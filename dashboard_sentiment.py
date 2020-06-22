import pandas as pd
import pickle
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table 
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import nltk
import re

df_sentiment = pickle.load(open('sentiment_words.sav', 'rb'))
df_sentiment['Sentiment'] = df_sentiment['Sentiment'].map({2:'Positive', 1 : 'Neutral', 0: 'Negative'})

model_title = pickle.load(open('model_title.sav', 'rb'))
model_review = pickle.load(open('model_review.sav', 'rb'))
model_combination = pickle.load(open('model_combination.sav', 'rb'))

def clean(data):    
    data = re.sub('[^a-zA-Z]', ' ', data)
    words = data.lower()
    return words

def show_most_word(sentiment, column, number = 20, n_grams=1, common='Common'):
    word = []
    
    # Split word based on the parameter
    if n_grams == 1:
        df_sentiment[df_sentiment['Sentiment'] == sentiment][column].apply(lambda x: word.extend((x.split())))
    elif n_grams == 2:
        df_sentiment[df_sentiment['Sentiment'] == sentiment][column].apply(lambda x: word.extend(nltk.bigrams(x.split())))
        
    # Use NLTK FreqDist to get most common word    
    if common == 'Common':
        words_count = nltk.FreqDist(word)
        most_common = words_count.most_common(number)
        dict_most = dict(most_common)
    elif common == 'Rare':    
        words_count = nltk.FreqDist(word)
        most_common = words_count.most_common()[-number:]
        dict_most = dict(most_common)
        
    # Plot the most common word
    if n_grams == 1:
        key = list(dict_most.keys())
    elif n_grams == 2:
        keys = list(dict_most.keys())
        b = []
        for i in range(len(keys)):
            c = ' '.join(keys[i])
            b.append(c)
        key = b   

    return key, dict_most     

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

def generate_table(dataframe, page_size=10, s='', column='alpha_title'):
    if s == '':
        dataframe = dataframe
    else:
        dataframe =  dataframe[dataframe['Sentiment'] == s]
    return dash_table.DataTable(
        id='dataTable',
        columns=[{
            "name": i,
            "id": i
        } for i in [column, 'Sentiment']],
        data=dataframe.to_dict('records'),
        page_action="native",
        page_current=0,
        page_size=page_size,
    )

key, dict_most = show_most_word('Positive', 'alpha_title', number = 20, n_grams=1, common='Common')

app.layout = html.Div(children=[
    html.H1('Final Project Dashboard'),
    html.P('Created by: Ivan Sebastian'),
    dcc.Tabs(children = [
        dcc.Tab(value = 'Tab1', label = 'DataFrame Table', children = [
            html.Center(html.H1('DATAFRAME')),
            html.Div([

            html.Div([
                html.P('Sentiment'),
                dcc.Dropdown(value = '',
                id = 'filter-site',
                options = [{'label' : 'All', 'value': ''},
                          {'label' : 'Positive', 'value': 'Positive'},
                          {'label' : 'Neutral', 'value': 'Neutral'},
                          {'label' : 'Negative', 'value': 'Negative'}])
            ], className = 'col-3'),
            html.Div([
                html.P('Column'),
                dcc.Dropdown(value = 'alpha_title',
                id = 'filter-column',
                options = [{'label' : 'Alphabet Title', 'value': 'alpha_title'},
                          {'label' : 'Alphabet Review', 'value': 'alpha_review'},
                          {'label' : 'Stemmed Title', 'value': 'stem_title'},
                          {'label' : 'Stemmed Review', 'value': 'stem_review'},
                          {'label' : 'Lemmatized Title', 'value': 'lemma_title'},
                          {'label' : 'Lemmatized Review', 'value': 'lemma_review'}])
            ], className = 'col-3'),
            html.Br(),
            html.Div([
                html.P('Max Row'),
                dcc.Input(id='filter-row', type='number', value = 10)
                ], className = 'col-3'),
            html.Br(),
            html.Div(children = [
                html.Button('Search', id='filter')
            ], className = 'col-4'),
            html.Br(),
            html.Div(id = 'div_table',
            children = [generate_table(df_sentiment, 10, 'alpha_title')])
            ]
            )]
    ),
    dcc.Tab(label = 'Visualization', children = [
            html.Div(children=[
                html.Div([
                    html.P('Column:'),
                    dcc.Dropdown(
                        id = 'columnbarplot',
                        options = [{'label': 'Alphabet Title', 'value': 'alpha_title'},
                                    {'label': 'Alphabet Review', 'value': 'alpha_review'},
                                    {'label': 'Stemmed Title', 'value': 'stem_title'},
                                    {'label': 'Stemmed Review', 'value': 'stem_review'},
                                    {'label': 'Lemmatized Title', 'value': 'lemma_title'},
                                    {'label': 'Lemmatized Review', 'value': 'lemma_review'}],
                        value = 'alpha_title'
                    )], className = 'col-3'),
                html.Div([
                    html.P('N-gram:'),
                    dcc.Dropdown(
                        id = 'ngrambarplot',
                        options = [{'label': 'Uni-gram', 'value': 1},
                                    {'label': 'Bi-gram', 'value': 2}],
                        value = 1
                    )], className = 'col-3'),
                html.Div([
                    html.P('Sentiment:'),
                    dcc.Dropdown(
                        id = 'sentimentbarplot',
                        options = [{'label': 'Positive', 'value': 'Positive'},
                                    {'label': 'Neutral', 'value': 'Neutral'},
                                    {'label': 'Negative', 'value': 'Negative'}],
                        value = 'Positive'
                    )], className = 'col-3'),
                html.Div([
                    html.P('Common or Rare:'),
                    dcc.Dropdown(
                        id = 'commonbarplot',
                        options = [{'label': 'Common', 'value': 'Common'},
                                    {'label': 'Rare', 'value': 'Rare'}],
                        value = 'Common'
                    )], className = 'col-3')],
                    className = 'row'),
                html.Div([
                dcc.Graph(
                    id = 'barplot',
                    figure = {
                        'data' : [
                            go.Bar(
                                x = key,
                                y = list(dict_most.values()),
                                name = 'Plot'
                            )],
                            'layout':go.Layout(
                                title='20 Most Common Uni-gram Words in Positive alpha_title')
                    }
                )
                ])
    ]),
    dcc.Tab(label = 'Sentiment Prediction Model', value = 'tab=4', children=[
            html.Div([
                html.Div([
                    html.Center([
                        html.H2('Sentiment Prediction Model', className='title')])
                     ]),
                    html.Br(),
                html.Div([        
                    html.P('Insert Title: '),
                    dcc.Input(
                        id ='titlepredict',
                        type='text',
                        value = '',
                        style=dict(width='100%')
                    )], className = 'col-12 row'),
                    html.Br(),
                    html.Div([
                        html.P('Insert Review: '),
                    dcc.Textarea(
                    id = 'reviewpredict',
                    value='',
                    style={'width': '100%'} 
                        )], className = 'col-12 row'
                    ),
                    html.Br(),
                    html.Div([
                        html.Button('Predict', id='predict', style=dict(width='100%'))
                    ], className='col-2 row'),
                    html.Div([
                        html.Center([],id ='prediction')
                    ])]
                )]
                )
],
    content_style = {
        'fontFamily': 'Arial',
        'borderBottom': '1px solid #d6d6d6',
        'borderLeft': '1px solid #d6d6d6',
        'borderRight': '1px solid #d6d6d6',
        'padding': '44px'
    })],
        style = {
    'maxwidth' : '1200px',
    'margin': '0 auto'
}
)

@app.callback(
    Output(component_id = 'div_table', component_property = 'children'),
    [Input(component_id = 'filter', component_property = 'n_clicks')],
    [State(component_id = 'filter-row', component_property = 'value'),
    State(component_id = 'filter-site', component_property = 'value'),
    State(component_id = 'filter-column', component_property = 'value')])
    

def update_table(n_clicks, row, filtersite, filtercolumn):
    children = [generate_table(df_sentiment, row, filtersite, filtercolumn)]
    return children

@app.callback(
    Output(component_id = 'barplot', component_property = 'figure'),
    [Input(component_id = 'ngrambarplot', component_property = 'value'),
    Input(component_id = 'sentimentbarplot', component_property = 'value'),
    Input(component_id = 'commonbarplot', component_property = 'value'),
    Input(component_id = 'columnbarplot', component_property = 'value')]
)

def update_graph(ngrambarplot, sentimentbarplot, commonbarplot, columnbarplot):
    key, dict_most = show_most_word(sentimentbarplot, columnbarplot, number = 20, n_grams=ngrambarplot, common=commonbarplot)
    dict_gram = {1: 'Uni-gram', 2:'Bi-gram'}
    return{
        'data' : [
                    go.Bar(
                            x = key,
                            y = list(dict_most.values())
                            )],
                        'layout':go.Layout(
                            title='20 Most {} {} Words in {} {}'.format(commonbarplot, dict_gram[ngrambarplot], sentimentbarplot, columnbarplot))
                    }

@app.callback(
    Output(component_id='prediction', component_property='children'),
    [Input('predict', 'n_clicks')],
    [State('titlepredict', 'value'),
     State('reviewpredict', 'value')]
)

def model_predict(n_clicks, title_text, review_text):
    predict_dict = {0 : 'Negative', 1: 'Neutral', 2: 'Positive'}
    title_text = clean(str(title_text))
    review_text = clean(str(review_text))
    if (title_text == '') and (review_text == ''):
        return html.H4('No text in the box, please fill either Title or Review')
    elif (title_text != '') and (review_text == ''):
        title = model_title.predict([title_text])[0]
        title_prob = model_title.predict_proba([title_text])[0][title]
        return [html.H4('This title is classified as {} Sentiment text with probability {}%'.format(predict_dict[title], round(title_prob*100,2)))]    
    elif (title_text == '') and (review_text != ''):
        review = model_review.predict([review_text])[0]
        review_prob = model_review.predict_proba([review_text])[0][review]
        return [html.H4('This review is classified as {} Sentiment text with probability {}%'.format(predict_dict[review], round(review_prob*100,2)))]
    elif (title_text != '') and (review_text != ''): 
        title = model_title.predict([title_text])[0]
        title_prob = model_title.predict_proba([title_text])[0][title]
        review = model_review.predict([review_text])[0]
        review_prob = model_review.predict_proba([review_text])[0][review]
        combination_text = ' '.join([title_text, review_text])
        combination = model_combination.predict([combination_text])[0]
        combination_prob = model_combination.predict_proba([combination_text])[0][combination]
        return [html.H4('The Title is classified as {} Sentiment text with probability {}%'.format(predict_dict[title], round(title_prob*100,2))),
                html.H4('The Review is classified as {} Sentiment text with probability {}%'.format(predict_dict[review], round(review_prob*100,2))),
                html.H4('Combination of the Title and The Review is classified as {} Sentiment text with probability {}%'.format(predict_dict[combination], round(combination_prob*100)))]

if __name__ == '__main__':
    app.run_server(debug=True)