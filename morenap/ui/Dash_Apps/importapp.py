import dash
from dash import Dash, dcc, html, Input, Output, State
from django_plotly_dash import DjangoDash
import dash_bootstrap_components as dbc


import json


class saved:
    def __init__(self) -> None:
        self.selected_labels = []
        self.selected_classes = []


json_file = json.load(open("results/search.json"))
db = saved

# print(json_file.keys())
# [print(label, list(zip(a_class.keys(),a_class.values()))) for label, a_class in json_file.items()]

def gen_dropdown(name, options):
    options_list = [f"{key} ({value:0.4f})" for key, value in options.items()]
    return html.Div(
                style={
                    "display": "grid",
                    "grid-template-columns": "10fr 90fr",
                    "padding-bottom": "30px",
                },
                children = [
                    f"{name} :",
                    dcc.Dropdown(
                        id=f"dropdown_{name}", 
                        options=options_list,
                        multi=False,
                        clearable=False,
                        value=options_list[0],
                        searchable=True,
                    ),
                ])



app = DjangoDash('importapp')   # replaces dash.Dash
# app = DjangoDash(__name__)

app.layout = html.Div([
    html.Div([
        "Score Theshold: ",
        dcc.Input(id='score_theshold', type='number', value=0, min=0, max=1, step=0.01),
    ]),
    html.Div(id='container-button-basic',
             children='Select classes and press submit'),  
    html.Br(),
    html.Div(
        style={
            "display": "grid",
            "grid-template-columns": "10fr 90fr",
            "padding-bottom": "30px",},
        children=[
            html.Div(["label"]), 
            html.Div(["class"]),
        ],
    ),
    html.Div(id="dropboxes"),
    html.Button('Submit', id='submit-val', n_clicks=0),
    html.Div(id="test_text"),
    dcc.Store(id="labels")
])


@app.callback(
    # [Output('test_text', 'children'),
    #  Output('dropboxes', 'children'),],
    Output("dropboxes", "children"),
    Output("labels", "data"),
    Input("score_theshold", "value"),
)
def select_theshold(score_theshold):
    a_dropbox = []
    _label_list = []
    for label, a_class in json_file.items():
        if list(a_class.values())[-1] >= float(score_theshold):
            a_dropbox.append(gen_dropdown(label, a_class))
            _label_list.append(label)
    return a_dropbox, _label_list

@app.callback(
    Output("test_text", "children"),
    Input("labels", "data")
    # [Input(f"dropdown_{label}", "value") for label in list(Input('dropboxes', 'children'))]
    # [Input(f"dropdown_{label}", "value") for label in db.selected_labels]
)
def select_classes(dropboxes_value):
    print(dropboxes_value)
    a_text = f"The input value was '{dropboxes_value}' and the button has been clicked times"
    return a_text

if __name__ == '__main__':
    app.run_server(debug=True)
