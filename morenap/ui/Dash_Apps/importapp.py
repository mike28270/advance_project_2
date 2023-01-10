import dash
from dash import Dash, dcc, html, Input, Output, State
from django_plotly_dash import DjangoDash
import dash_bootstrap_components as dbc


import json

json_file = json.load(open("results/search.json"))
print(json_file.keys())
[print(label, list(zip(a_class.keys(),a_class.values()))) for label, a_class in json_file.items()]

def gen_dropdown(name, options):
    return html.Div(
        className='div-for-dropdown',
        children=[
            dcc.Dropdown(
                id=f"dropdown_{name}", 
                options=options,
                # multi=True,
                value=1,
                # style={'backgroundColor': '#1E1E1E'},
                className='stockselector'
            ),
        ],
    )

app = DjangoDash('importapp')   # replaces dash.Dash
# app = DjangoDash(__name__)

app.layout = html.Div([
    html.Div([
        "Input: ",
        dcc.Input(id='range', type='number', min=0, max=1, step=0.01),
    ]),
    html.Button('Submit', id='submit-val', n_clicks=0),
    html.Div(id='container-button-basic',
             children='Enter a value and press submit'),    
    html.Div(children=[gen_dropdown(label, list(a_class.keys())) for label, a_class in json_file.items()]),

])


@app.callback(
    Output('container-button-basic', 'children'),
    Input('submit-val', 'n_clicks'),
    State('dropdown1', 'value')
)
def update_output(n_clicks, value):
    return 'The input value was "{}" and the button has been clicked {} times'.format(
        value,
        n_clicks
    )


if __name__ == '__main__':
    app.run_server(debug=True)
