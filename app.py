import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("city_day.csv")
df.drop(columns=['Xylene'], inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df['Year']       = df['Date'].dt.year
df['Month']      = df['Date'].dt.month
df['Month_Name'] = df['Date'].dt.strftime('%b')
df['Season']     = df['Month'].map({
    12:'Winter', 1:'Winter', 2:'Winter',
     3:'Spring', 4:'Spring', 5:'Spring',
     6:'Summer', 7:'Summer', 8:'Summer',
     9:'Autumn',10:'Autumn',11:'Autumn'
})
fill_cols = ['PM2.5','PM10','NO','NO2','NOx','NH3',
             'CO','SO2','O3','Benzene','Toluene']
for col in fill_cols:
    df[col] = df.groupby('City')[col].transform(lambda x: x.fillna(x.median()))
    df[col] = df[col].fillna(df[col].median())
df.dropna(subset=['AQI','AQI_Bucket'], inplace=True)
df.reset_index(drop=True, inplace=True)

BG        = '#0d1117'
CARD      = '#161b27'
CARD2     = '#1e2535'
TEAL      = '#00d4aa'
RED       = '#ff4d4d'
ORANGE    = '#ff8c42'
YELLOW    = '#ffd166'
TEXT      = '#e6edf3'
SUBTEXT   = '#8b949e'
BORDER    = '#30363d'
FONT      = 'Rajdhani, sans-serif'

AQI_COLORS = {
    'Good'        : '#00e676',
    'Satisfactory': '#aeea00',
    'Moderate'    : '#ffd600',
    'Poor'        : '#ff9100',
    'Very Poor'   : '#ff3d00',
    'Severe'      : '#b71c1c',
}

def card(children, style={}):
    base = {
        'background'  : CARD,
        'borderRadius': '16px',
        'padding'     : '24px',
        'border'      : f'1px solid {BORDER}',
        'boxShadow'   : '0 4px 24px rgba(0,0,0,0.4)',
    }
    base.update(style)
    return html.Div(children, style=base)

def section_title(text):
    return html.Div([
        html.Span(text, style={
            'fontSize'    : '13px',
            'fontWeight'  : '700',
            'letterSpacing': '3px',
            'color'       : TEAL,
            'textTransform': 'uppercase',
            'fontFamily'  : FONT,
        })
    ], style={'marginBottom': '16px'})


fig_aqi_dist = go.Figure()
fig_aqi_dist.add_trace(go.Histogram(
    x            = df['AQI'],
    nbinsx       = 60,
    marker_color = TEAL,
    opacity      = 0.85,
    name         = 'AQI Frequency',
    hovertemplate= 'AQI Range: %{x}<br>Days: %{y}<extra></extra>',
))
fig_aqi_dist.add_vline(
    x=df['AQI'].mean(), line_dash='dash',
    line_color=ORANGE, line_width=2,
    annotation_text=f"Mean: {df['AQI'].mean():.0f}",
    annotation_font_color=ORANGE,
    annotation_font_size=12,
)
fig_aqi_dist.update_layout(
    paper_bgcolor = CARD,
    plot_bgcolor  = CARD,
    font          = dict(family=FONT, color=TEXT),
    margin        = dict(l=40, r=20, t=20, b=40),
    xaxis = dict(
        title      = 'AQI Value',
        gridcolor  = BORDER,
        showline   = True,
        linecolor  = BORDER,
        tickfont   = dict(color=SUBTEXT),
    ),
    yaxis = dict(
        title      = 'Number of Days',
        gridcolor  = BORDER,
        showline   = True,
        linecolor  = BORDER,
        tickfont   = dict(color=SUBTEXT),
    ),
    hoverlabel = dict(
        bgcolor   = CARD2,
        font_size = 13,
        font_family = FONT,
    ),
    bargap = 0.05,
)


app = Dash(__name__, title="India AQI Dashboard")
app.layout = html.Div(style={
    'backgroundColor': BG,
    'minHeight'      : '100vh',
    'fontFamily'     : FONT,
    'padding'        : '0',
}, children=[

    html.Link(
        rel  = 'stylesheet',
        href = 'https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&display=swap'
    ),

    html.Div(style={
        'background'  : f'linear-gradient(135deg, #0d2137 0%, #0d1117 60%)',
        'borderBottom': f'1px solid {BORDER}',
        'padding'     : '32px 48px',
        'position'    : 'relative',
        'overflow'    : 'hidden',
    }, children=[
        html.Div(style={
            'position'  : 'absolute', 'top': '0', 'left': '0',
            'right'     : '0', 'bottom': '0',
            'background': f'radial-gradient(ellipse at 20% 50%, rgba(0,212,170,0.07) 0%, transparent 60%)',
        }),
        html.Div([
            html.Div(style={'display':'flex','alignItems':'center','gap':'16px'}, children=[
                html.Div(style={
                    'width':'48px','height':'48px','borderRadius':'12px',
                    'background':f'linear-gradient(135deg,{TEAL},{TEAL}88)',
                    'display':'flex','alignItems':'center','justifyContent':'center',
                    'fontSize':'24px',
                }, children='🌫'),
                html.Div([
                    html.H1("India Air Quality Dashboard", style={
                        'margin':'0','fontSize':'28px','fontWeight':'700',
                        'color':TEXT,'letterSpacing':'1px','fontFamily':FONT,
                    }),
                    html.P("EDA & AQI Prediction — 26 Cities | 2015–2020", style={
                        'margin':'4px 0 0','fontSize':'13px',
                        'color':SUBTEXT,'fontFamily':FONT,'letterSpacing':'1px',
                    }),
                ]),
            ]),
        ]),
    ]),

    html.Div(style={'padding':'32px 48px'}, children=[

        html.Div(style={
            'display':'grid',
            'gridTemplateColumns':'repeat(4, 1fr)',
            'gap':'16px',
            'marginBottom':'32px',
        }, children=[
            card([
                html.P("Total Records", style={'margin':'0','fontSize':'11px','color':SUBTEXT,'letterSpacing':'2px','textTransform':'uppercase'}),
                html.H2(f"{len(df):,}", style={'margin':'8px 0 0','fontSize':'32px','fontWeight':'700','color':TEAL}),
                html.P("cleaned data points", style={'margin':'4px 0 0','fontSize':'11px','color':SUBTEXT}),
            ]),
            card([
                html.P("Cities Monitored", style={'margin':'0','fontSize':'11px','color':SUBTEXT,'letterSpacing':'2px','textTransform':'uppercase'}),
                html.H2(f"{df['City'].nunique()}", style={'margin':'8px 0 0','fontSize':'32px','fontWeight':'700','color':TEAL}),
                html.P("across India", style={'margin':'4px 0 0','fontSize':'11px','color':SUBTEXT}),
            ]),
            card([
                html.P("Average AQI", style={'margin':'0','fontSize':'11px','color':SUBTEXT,'letterSpacing':'2px','textTransform':'uppercase'}),
                html.H2(f"{df['AQI'].mean():.0f}", style={'margin':'8px 0 0','fontSize':'32px','fontWeight':'700','color':ORANGE}),
                html.P("moderate-poor range", style={'margin':'4px 0 0','fontSize':'11px','color':SUBTEXT}),
            ]),
            card([
                html.P("Peak AQI Recorded", style={'margin':'0','fontSize':'11px','color':SUBTEXT,'letterSpacing':'2px','textTransform':'uppercase'}),
                html.H2(f"{df['AQI'].max():.0f}", style={'margin':'8px 0 0','fontSize':'32px','fontWeight':'700','color':RED}),
                html.P("severe hazard level", style={'margin':'4px 0 0','fontSize':'11px','color':SUBTEXT}),
            ]),
        ]),

        card([
            section_title("Chart 1 — AQI Distribution Across All Cities & Years"),
            html.P(
                "Distribution of daily AQI values recorded across 26 Indian cities from 2015 to 2020. "
                "The orange dashed line marks the overall mean AQI of 166.",
                style={'fontSize':'13px','color':SUBTEXT,'marginBottom':'16px','fontFamily':FONT}
            ),
            dcc.Graph(figure=fig_aqi_dist, config={'displayModeBar':False}, style={'height':'380px'}),
        ], style={'marginBottom':'24px'}),

    ]),
])

if __name__ == '__main__':
    app.run(debug=True)