import dash
from dash import Dash, dcc, html, Input, Output, State
from django_plotly_dash import DjangoDash
import dash_bootstrap_components as dbc


import json


json_file = json.load(open("results/search.json"))
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
                        # multi=True,
                        value=1,
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
    html.Div(id="container-button-basic")
])


@app.callback(
    [Output('container-button-basic', 'children'),
     Output('dropboxes', 'children'),],
    Input('score_theshold', 'value')
)
def select_theshold(score_theshold):
    a_text = f"The input value was '{score_theshold}' and the button has been clicked times"
    a_dropbox = []
    for label, a_class in json_file.items():
        # print(list(a_class.values())[0])
        if list(a_class.values())[0] >= float(score_theshold):
            a_dropbox.append(gen_dropdown(label, a_class))
    return [a_text, a_dropbox]

# @app.callback(
#     Output('container-button-basic', 'children'),
#     State('dropboxes', 'value')
# )
# def select_classes(dropboxes):
#     pass

if __name__ == '__main__':
    app.run_server(debug=True)
