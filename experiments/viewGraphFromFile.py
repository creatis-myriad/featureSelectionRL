import json

import dash
import dash_cytoscape as cyto
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output
import pickle,argparse, io, numpy as np
import graphStructure
from dash.exceptions import PreventUpdate
from matplotlib import cm, colors


parser = argparse.ArgumentParser(description='Display a graph.')
parser.add_argument('path', help='path of the .pkl file')
parser.add_argument('--showOnlyVisited', default = False, action = 'store_true', help = 'Display only nodes visited at least once')
parser.add_argument('--thresholdVisited', default = 0)

parser.add_argument('--plotS0', default = False, action = 'store_true', help  = 'Display also starting node')

args = parser.parse_args()


class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "experiments.graphStructure":
            renamed_module = "graphStructure"
        return super(RenameUnpickler, self).find_class(renamed_module, name)

def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()

with open(args.path, 'rb') as f:
    graph = renamed_load(f)

if args.showOnlyVisited:
    graph.filterByVisits(args.thresholdVisited)

# enable svg export
cyto.load_extra_layouts()
app = dash.Dash(__name__)

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}


viridis = cm.get_cmap('Blues', 12)

nodes_main = [
    {
        'data': {'id': str(k), 
                 'label':  '{' +  ', '.join(graph.nodes[k]) + '}' + '\n' + f' #Visited={np.sum(graph.nodesVisited[k])}, #Finished={np.sum(graph.nodesFinished[k])}',
                  'percentage' : .5 + .5 *np.sum(graph.nodesVisited[k])/len(graph.nodesVisited[k]),
                  'color' :  colors.to_hex(viridis(.05 + .55 * np.sum(graph.nodesVisited[k])/len(graph.nodesVisited[k]))),
                },
        'classes' : 'mainNode'
    }
    for k in graph.nodesVisited if (args.plotS0 or k !=0)
]
for n in nodes_main:
    s =  n['data']['label'].split('\n')
    l1 = len(s[0].strip())
    l2 = len(s[1].strip())
    n['data']['length'] = max(l1, l2) * 9.5 + 70
    
#nodes_child = [
#    {
#        'data': {'id': str(k) +'_child', 'parent' : str(k), 
#                 'label2' : f'#Visited={np.sum(graph.nodesVisited[k])}, #Decision={np.sum(graph.nodesFinished[k])}',
#                 'percentage' : .5 + .5 * np.sum(graph.nodesVisited[k])/len(graph.nodesVisited[k]),
#                'color' :  colors.to_hex(viridis(np.sum(graph.nodesVisited[k])/len(graph.nodesVisited[k])))
#
#                },
#        'classes' : 'childNode'
#    }
#    for k in graph.nodesVisited if (args.plotS0 or k !=0)
#]


edges = [
    {'data': {'source': str(k1), 
              'target': str(k2),
              'label' : f'N = {np.sum(idx)}',
            'percentage' : 1 + 10 * (np.sum(idx)/len(idx))

             }
    
    }
    for (k1, k2), idx in graph.edgesVisited.items() if (args.plotS0 or k1 !=0)
]


default_stylesheet = [
    {
        'selector': 'node',
        'style': {
            'background-color': 'data(color)',
                'width': "data(length)",
                'height': 60,
                'shape':'roundrectangle',
                'text-wrap': 'wrap',
                'font-size' : 24

        }
    },
    {
        'selector' : '.childNode',
        'style': {
           'content': 'data(label2)',
            'background-opacity': 0,


        }
    },
    {
        'selector' : '.mainNode',
        'style': {
           'content': 'data(label)',
            'text-halign':'center',
            'text-valign':'center',
            'border-width': '1px',
            'border-color': 'black'

            }
    },

    {
        'selector': 'edge',
        'style': {
            'label': 'data(label)',
            'source-arrow-color': 'lightgrey',
            'target-arrow-shape': 'triangle',
            'line-color': 'lightgrey',
            'curve-style': 'straight',
            'width' : 'data(percentage)',
            'font-size' : 24

        }
    }
]


app.layout = html.Div([
    html.Div(className='four columns', children=[
        html.Div('Download graph:'),
            html.Button("as jpg", id="btn-get-jpg"),
            html.Button("as png", id="btn-get-png"),
            html.Button("as svg", id="btn-get-svg")
    ]),
    cyto.Cytoscape(
        id='cytoscape-graph',
        elements=edges+nodes_main, #+nodes_child,
        stylesheet=default_stylesheet,
        style={'width': '100%', 'height': '1200px'},
        layout={'name': 'dagre'}
        
    ),
    dcc.Tabs([
        dcc.Tab(label='Node', children=[
        html.P(id='cytoscape-tapNodeData-output'),
            ]),
        dcc.Tab(label='Edge', children=[
        html.P(id='cytoscape-tapEdgeData-output'),
            ]),
        
    ])
])

if False:
    @app.callback(Output('cytoscape-tapNodeData-output', 'children'),
                  Input('cytoscape-event-callbacks-2', 'tapNodeData'))
    def displayTapNodeData(nodeData):
        if nodeData:
            data = nodeData
            node = int(data['id'])
            data = graph['nodes'][node]
            data_reformatted = [
                {'var' : k, 'value' : str(v)}
                for k, v in data.items() 

            ]
            return dash_table.DataTable(
                            id='table-measurements',
                            columns = [{'name': v.upper(), 'id' : v} for v in ['var', 'value']],
                            data = data_reformatted
                        )
    @app.callback(Output('cytoscape-tapEdgeData-output', 'children'),
                  Input('cytoscape-event-callbacks-2', 'tapEdgeData'))
    def displayTapNodeData(edgeData):
        if edgeData:
            data = edgeData
            edge = (int(data['source']),int(data['target']))
            data = graph['edges'][edge]
            data_reformatted = [
                {'var' : k, 'value' : str(v)}
                for k, v in data.items() 

            ]
            return dash_table.DataTable(
                            id='table-measurements',
                            columns = [{'name': v.upper(), 'id' : v} for v in ['var', 'value']],
                            data = data_reformatted
                        )
        
@app.callback(
    Output("cytoscape-graph", "generateImage"),
    [
        Input("btn-get-jpg", "n_clicks"),
        Input("btn-get-png", "n_clicks"),
        Input("btn-get-svg", "n_clicks"),
    ])
def get_image(get_jpg_clicks, get_png_clicks, get_svg_clicks):

    # File type to output of 'svg, 'png', 'jpg', or 'jpeg' (alias of 'jpg')

    # 'store': Stores the image data in 'imageData' !only jpg/png are supported
    # 'download'`: Downloads the image as a file with all data handling
    # 'both'`: Stores image data and downloads image as file.

    ctx = dash.callback_context
    if ctx.triggered:
        input_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if input_id != "tabs":
            action = "download"
            ftype = input_id.split("-")[-1]

        return {
            'type': ftype,
            'action': action
            }
    else:
        raise PreventUpdate

if __name__ == '__main__':
    app.run_server(debug=True)