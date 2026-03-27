import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# ── Data Loading & Cleaning ─────────────────────────────────────
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
fill_cols = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene']
for col in fill_cols:
    df[col] = df.groupby('City')[col].transform(lambda x: x.fillna(x.median()))
    df[col] = df[col].fillna(df[col].median())
df.dropna(subset=['AQI','AQI_Bucket'], inplace=True)
df.reset_index(drop=True, inplace=True)

# ── AQI Helpers ─────────────────────────────────────────────────
def aqi_label(val):
    if val <= 50:   return "Good"
    if val <= 100:  return "Satisfactory"
    if val <= 200:  return "Moderate"
    if val <= 300:  return "Poor"
    if val <= 400:  return "Very Poor"
    return "Severe"

def aqi_gradient(val):
    if val <= 50:   return ("135deg", "#a8edea", "#fed6e3", "#b8f0c8")
    if val <= 100:  return ("135deg", "#ffecd2", "#fcb69f", "#ffe8a3")
    if val <= 200:  return ("135deg", "#ffd89b", "#ff9a5c", "#ffb347")
    if val <= 300:  return ("135deg", "#f093fb", "#f5576c", "#e96c7a")
    if val <= 400:  return ("135deg", "#c471ed", "#f64f59", "#b44fc4")
    return           ("135deg", "#4b0082", "#8b0000", "#6a0572")

def aqi_dot_color(val):
    if val <= 50:   return "#4ade80"
    if val <= 100:  return "#a3e635"
    if val <= 200:  return "#facc15"
    if val <= 300:  return "#fb923c"
    if val <= 400:  return "#f87171"
    return "#c084fc"

city_avg = df.groupby('City')['AQI'].mean().reset_index()
city_avg.columns = ['City','Avg_AQI']
city_avg['Label'] = city_avg['Avg_AQI'].apply(aqi_label)
city_avg = city_avg.sort_values('Avg_AQI', ascending=False).reset_index(drop=True)

overall_avg  = df['AQI'].mean()
overall_label = aqi_label(overall_avg)
worst_city   = city_avg.iloc[0]['City']
best_city    = city_avg.iloc[-1]['City']

# ── Chart Builders ──────────────────────────────────────────────
def make_chart(bg, border, text, subtext, grid, font):

    # Chart 1 – AQI Distribution
    fig1 = go.Figure()
    fig1.add_trace(go.Histogram(
        x=df['AQI'], nbinsx=55,
        marker=dict(
            color=df['AQI'].value_counts(bins=55, sort=False).index.mid,
            colorscale=[[0,'#4ade80'],[0.3,'#facc15'],[0.6,'#fb923c'],[1,'#f87171']],
            line=dict(width=0),
        ),
        opacity=0.85,
        hovertemplate='AQI: %{x}<br>Days: %{y}<extra></extra>',
    ))
    fig1.add_vline(x=overall_avg, line_dash='dot', line_color='white',
                   line_width=1.5,
                   annotation_text=f"Mean {overall_avg:.0f}",
                   annotation_font_color='white', annotation_font_size=11)
    fig1.update_layout(
        paper_bgcolor=bg, plot_bgcolor=bg,
        font=dict(family=font, color=text),
        margin=dict(l=36,r=12,t=12,b=36),
        xaxis=dict(title='AQI Value', gridcolor=grid, zeroline=False,
                   tickfont=dict(color=subtext, size=11)),
        yaxis=dict(title='Days', gridcolor=grid, zeroline=False,
                   tickfont=dict(color=subtext, size=11)),
        showlegend=False, bargap=0.04,
        hoverlabel=dict(bgcolor=border, font_size=12, font_family=font),
    )

    # Chart 2 – City AQI bar
    fig2 = go.Figure()
    colors2 = [aqi_dot_color(v) for v in city_avg['Avg_AQI']]
    fig2.add_trace(go.Bar(
        x=city_avg['City'], y=city_avg['Avg_AQI'],
        marker=dict(color=colors2, line=dict(width=0)),
        hovertemplate='%{x}<br>Avg AQI: %{y:.0f}<extra></extra>',
    ))
    fig2.update_layout(
        paper_bgcolor=bg, plot_bgcolor=bg,
        font=dict(family=font, color=text),
        margin=dict(l=36,r=12,t=12,b=80),
        xaxis=dict(tickangle=-40, gridcolor=grid, zeroline=False,
                   tickfont=dict(color=subtext, size=10)),
        yaxis=dict(gridcolor=grid, zeroline=False,
                   tickfont=dict(color=subtext, size=11)),
        showlegend=False,
        hoverlabel=dict(bgcolor=border, font_size=12, font_family=font),
    )

    # Chart 3 – Monthly seasonality
    month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    monthly = df.groupby('Month_Name')['AQI'].mean().reindex(month_order).reset_index()
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=monthly['Month_Name'], y=monthly['AQI'],
        mode='lines+markers',
        line=dict(color='#a78bfa', width=3),
        marker=dict(size=8, color=[aqi_dot_color(v) for v in monthly['AQI']],
                    line=dict(width=2, color=bg)),
        fill='tozeroy',
        fillcolor='rgba(167,139,250,0.12)',
        hovertemplate='%{x}<br>Avg AQI: %{y:.0f}<extra></extra>',
    ))
    fig3.update_layout(
        paper_bgcolor=bg, plot_bgcolor=bg,
        font=dict(family=font, color=text),
        margin=dict(l=36,r=12,t=12,b=36),
        xaxis=dict(gridcolor=grid, zeroline=False,
                   tickfont=dict(color=subtext, size=11)),
        yaxis=dict(gridcolor=grid, zeroline=False,
                   tickfont=dict(color=subtext, size=11)),
        showlegend=False,
        hoverlabel=dict(bgcolor=border, font_size=12, font_family=font),
    )

    # Chart 4 – Correlation heatmap
    corr_cols = ['PM2.5','PM10','NO2','CO','SO2','O3','Benzene','AQI']
    corr = df[corr_cols].corr().round(2)
    fig4 = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale=[[0,'#1e3a5f'],[0.5,'#a78bfa'],[1,'#f87171']],
        text=corr.values, texttemplate='%{text}',
        textfont=dict(size=11, color=text),
        hovertemplate='%{x} × %{y}<br>r = %{z}<extra></extra>',
        showscale=True,
        colorbar=dict(tickfont=dict(color=subtext), outlinewidth=0),
    ))
    fig4.update_layout(
        paper_bgcolor=bg, plot_bgcolor=bg,
        font=dict(family=font, color=text),
        margin=dict(l=80,r=12,t=12,b=80),
        xaxis=dict(tickangle=-35, tickfont=dict(color=subtext, size=10)),
        yaxis=dict(tickfont=dict(color=subtext, size=10)),
        hoverlabel=dict(bgcolor=border, font_size=12, font_family=font),
    )

    # Chart 5 – PM2.5 vs AQI scatter
    sample = df.sample(2500, random_state=42)
    dot_colors = [aqi_dot_color(v) for v in sample['AQI']]
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=sample['PM2.5'], y=sample['AQI'],
        mode='markers',
        marker=dict(color=dot_colors, size=5, opacity=0.55,
                    line=dict(width=0)),
        hovertemplate='PM2.5: %{x:.1f}<br>AQI: %{y:.0f}<extra></extra>',
    ))
    fig5.update_layout(
        paper_bgcolor=bg, plot_bgcolor=bg,
        font=dict(family=font, color=text),
        margin=dict(l=36,r=12,t=12,b=36),
        xaxis=dict(title='PM2.5', gridcolor=grid, zeroline=False,
                   tickfont=dict(color=subtext, size=11)),
        yaxis=dict(title='AQI', gridcolor=grid, zeroline=False,
                   tickfont=dict(color=subtext, size=11)),
        showlegend=False,
        hoverlabel=dict(bgcolor=border, font_size=12, font_family=font),
    )

    # Chart 6 – Year trend top 5 cities
    top5 = city_avg.head(5)['City'].tolist()
    yearly = df[df['City'].isin(top5)].groupby(['Year','City'])['AQI'].mean().reset_index()
    palette6 = ['#f87171','#fb923c','#facc15','#a78bfa','#4ade80']
    fig6 = go.Figure()
    for i, city in enumerate(top5):
        d = yearly[yearly['City']==city]
        fig6.add_trace(go.Scatter(
            x=d['Year'], y=d['AQI'], name=city,
            mode='lines+markers',
            line=dict(color=palette6[i], width=2.5),
            marker=dict(size=7, color=palette6[i]),
            hovertemplate=f'{city}<br>Year: %{{x}}<br>AQI: %{{y:.0f}}<extra></extra>',
        ))
    fig6.update_layout(
        paper_bgcolor=bg, plot_bgcolor=bg,
        font=dict(family=font, color=text),
        margin=dict(l=36,r=12,t=12,b=36),
        xaxis=dict(gridcolor=grid, zeroline=False, dtick=1,
                   tickfont=dict(color=subtext, size=11)),
        yaxis=dict(gridcolor=grid, zeroline=False,
                   tickfont=dict(color=subtext, size=11)),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=subtext, size=11)),
        hoverlabel=dict(bgcolor=border, font_size=12, font_family=font),
    )

    return fig1, fig2, fig3, fig4, fig5, fig6


# ── App Init ────────────────────────────────────────────────────
app = Dash(__name__, title="Breath of the City")

# ── City Cards ─────────────────────────────────────────────────
def city_card(row):
    val   = row['Avg_AQI']
    label = row['Label']
    city  = row['City']
    ang, c1, c2, c3 = aqi_gradient(val)
    dot   = aqi_dot_color(val)
    return html.Div([
        html.Div(style={
            'background' : f'linear-gradient({ang},{c1},{c2})',
            'borderRadius': '20px',
            'padding'    : '24px 20px',
            'cursor'     : 'pointer',
            'transition' : 'transform 0.3s ease, box-shadow 0.3s ease',
            'boxShadow'  : '0 4px 20px rgba(0,0,0,0.12)',
            'position'   : 'relative',
            'overflow'   : 'hidden',
            'minHeight'  : f'{110 + (val/12):.0f}px',
        }, children=[
            html.Div(style={
                'position':'absolute','top':'-20px','right':'-20px',
                'width':'80px','height':'80px','borderRadius':'50%',
                'background':'rgba(255,255,255,0.15)',
            }),
            html.Div(style={
                'position':'absolute','bottom':'-30px','left':'10px',
                'width':'100px','height':'100px','borderRadius':'50%',
                'background':'rgba(255,255,255,0.08)',
            }),
            html.Div([
                html.Div(style={
                    'display':'flex','justifyContent':'space-between',
                    'alignItems':'flex-start','marginBottom':'16px',
                }, children=[
                    html.Span(city, style={
                        'fontSize':'14px','fontWeight':'600',
                        'color':'rgba(0,0,0,0.7)',
                        'letterSpacing':'0.3px',
                    }),
                    html.Span(style={
                        'width':'10px','height':'10px','borderRadius':'50%',
                        'background':dot,'display':'inline-block',
                        'boxShadow':f'0 0 8px {dot}',
                    }),
                ]),
                html.Div(f"{val:.0f}", style={
                    'fontSize':'42px','fontWeight':'700',
                    'color':'rgba(0,0,0,0.75)',
                    'lineHeight':'1','marginBottom':'8px',
                }),
                html.Span(label, style={
                    'fontSize':'11px','fontWeight':'600',
                    'letterSpacing':'2px','textTransform':'uppercase',
                    'color':'rgba(0,0,0,0.5)',
                    'background':'rgba(255,255,255,0.35)',
                    'borderRadius':'20px','padding':'3px 10px',
                }),
            ]),
        ]),
    ], style={'breakInside':'avoid','marginBottom':'16px'})


# ── Layout ──────────────────────────────────────────────────────
app.layout = html.Div(id='root', children=[

    html.Link(rel='stylesheet',
              href='https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap'),

    html.Link(rel='stylesheet', href='data:text/css,' + '''
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: "Poppins", sans-serif; }
        .card-hover:hover {
            transform: translateY(-6px) !important;
            box-shadow: 0 16px 40px rgba(0,0,0,0.18) !important;
        }
        .glass {
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
        }
        .nav-link { transition: opacity 0.2s; }
        .nav-link:hover { opacity: 0.6; }
        .toggle-btn {
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .toggle-btn:hover { opacity: 0.8; }
        @keyframes breathe {
            0%,100% { transform: scale(1); opacity: 0.6; }
            50% { transform: scale(1.18); opacity: 0.9; }
        }
        @keyframes breathe2 {
            0%,100% { transform: scale(1); opacity: 0.3; }
            50% { transform: scale(1.35); opacity: 0.5; }
        }
        @keyframes float {
            0%,100% { transform: translateY(0px) translateX(0px); opacity:0; }
            10% { opacity: 0.6; }
            90% { opacity: 0.4; }
            100% { transform: translateY(-120px) translateX(30px); opacity:0; }
        }
        @keyframes fadeUp {
            from { opacity:0; transform:translateY(20px); }
            to   { opacity:1; transform:translateY(0); }
        }
        .animate-fade-up { animation: fadeUp 0.7s ease forwards; }
        .masonry {
            columns: 4;
            column-gap: 16px;
        }
        @media (max-width: 1200px) { .masonry { columns: 3; } }
        @media (max-width: 800px)  { .masonry { columns: 2; } }
    '''),

    dcc.Store(id='mode-store', data='clean'),

    html.Div(id='page-wrapper', style={
        'minHeight':'100vh',
        'transition':'background 0.8s ease',
    }, children=[

        # Floating particles
        html.Div(id='particles', style={'position':'fixed','inset':'0','pointerEvents':'none','zIndex':'0'}, children=[
            *[html.Div(style={
                'position':'absolute',
                'width' : f'{np.random.randint(4,12)}px',
                'height': f'{np.random.randint(4,12)}px',
                'borderRadius':'50%',
                'background':'rgba(255,255,255,0.4)',
                'left'  : f'{np.random.randint(5,95)}%',
                'top'   : f'{np.random.randint(20,100)}%',
                'animation': f'float {np.random.randint(6,14)}s {np.random.randint(0,8)}s infinite ease-in-out',
            }) for _ in range(22)]
        ]),

        # Navbar
        html.Nav(id='navbar', className='glass', style={
            'position'  : 'sticky','top':'0','zIndex':'100',
            'padding'   : '0 48px',
            'height'    : '64px',
            'display'   : 'flex',
            'alignItems': 'center',
            'justifyContent':'space-between',
            'transition':'background 0.8s ease, border-color 0.8s ease',
        }, children=[
            html.Div(style={'display':'flex','alignItems':'center','gap':'10px'}, children=[
                html.Div('◎', style={'fontSize':'22px','color':'inherit'}),
                html.Span("Breath of the City", style={
                    'fontSize':'16px','fontWeight':'600','letterSpacing':'0.5px',
                }),
            ]),
            html.Div(style={'display':'flex','gap':'32px','alignItems':'center'}, children=[
                html.Span("Overview",    className='nav-link', style={'fontSize':'13px','fontWeight':'500','cursor':'pointer'}),
                html.Span("Cities",     className='nav-link', style={'fontSize':'13px','fontWeight':'500','cursor':'pointer'}),
                html.Span("Analytics",  className='nav-link', style={'fontSize':'13px','fontWeight':'500','cursor':'pointer'}),
                html.Div(id='toggle-btn', n_clicks=0, className='toggle-btn', style={
                    'display'   :'flex','alignItems':'center','gap':'8px',
                    'background':'rgba(255,255,255,0.2)',
                    'border'    :'1px solid rgba(255,255,255,0.3)',
                    'borderRadius':'20px','padding':'6px 14px',
                    'fontSize'  :'12px','fontWeight':'500',
                }),
            ]),
        ]),

        # Hero
        html.Div(id='hero', style={
            'position' :'relative','overflow':'hidden',
            'padding'  :'80px 48px 60px',
            'textAlign':'center',
            'zIndex'   :'1',
        }, children=[
            html.Div(style={'position':'relative','display':'inline-block','marginBottom':'32px'}, children=[
                html.Div(style={
                    'width':'180px','height':'180px','borderRadius':'50%',
                    'background':'rgba(255,255,255,0.12)',
                    'position':'absolute','top':'50%','left':'50%',
                    'transform':'translate(-50%,-50%)',
                    'animation':'breathe2 4s ease-in-out infinite',
                }),
                html.Div(style={
                    'width':'130px','height':'130px','borderRadius':'50%',
                    'background':'rgba(255,255,255,0.18)',
                    'display':'flex','alignItems':'center','justifyContent':'center',
                    'animation':'breathe 4s ease-in-out infinite',
                    'position':'relative','zIndex':'2',
                }, children=[
                    html.Div([
                        html.Div(f"{overall_avg:.0f}", id='hero-aqi', style={
                            'fontSize':'44px','fontWeight':'700','lineHeight':'1',
                        }),
                        html.Div("AQI", style={'fontSize':'13px','fontWeight':'400','opacity':'0.7','letterSpacing':'3px'}),
                    ]),
                ]),
            ]),
            html.H1("Breath of the City", className='animate-fade-up', style={
                'fontSize':'42px','fontWeight':'700',
                'letterSpacing':'-0.5px','marginBottom':'12px',
            }),
            html.P(
                f"Monitoring air quality across {df['City'].nunique()} Indian cities · 2015 – 2020",
                style={'fontSize':'15px','fontWeight':'300','opacity':'0.7','marginBottom':'32px'},
            ),
            html.Div(style={'display':'flex','justifyContent':'center','gap':'24px','flexWrap':'wrap'}, children=[
                html.Div([
                    html.Div(worst_city, style={'fontSize':'18px','fontWeight':'600'}),
                    html.Div("Most Polluted", style={'fontSize':'11px','opacity':'0.6','letterSpacing':'1px','textTransform':'uppercase','marginTop':'2px'}),
                ], style={
                    'background':'rgba(255,255,255,0.15)','borderRadius':'14px',
                    'padding':'14px 24px','backdropFilter':'blur(10px)',
                }),
                html.Div([
                    html.Div(best_city, style={'fontSize':'18px','fontWeight':'600'}),
                    html.Div("Cleanest City", style={'fontSize':'11px','opacity':'0.6','letterSpacing':'1px','textTransform':'uppercase','marginTop':'2px'}),
                ], style={
                    'background':'rgba(255,255,255,0.15)','borderRadius':'14px',
                    'padding':'14px 24px','backdropFilter':'blur(10px)',
                }),
                html.Div([
                    html.Div("November", style={'fontSize':'18px','fontWeight':'600'}),
                    html.Div("Most Polluted Month", style={'fontSize':'11px','opacity':'0.6','letterSpacing':'1px','textTransform':'uppercase','marginTop':'2px'}),
                ], style={
                    'background':'rgba(255,255,255,0.15)','borderRadius':'14px',
                    'padding':'14px 24px','backdropFilter':'blur(10px)',
                }),
            ]),
        ]),

        # Body content
        html.Div(style={'position':'relative','zIndex':'1','padding':'0 48px 60px'}, children=[

            # City Cards Grid
            html.Div(style={'marginBottom':'56px'}, children=[
                html.Div(style={'marginBottom':'24px'}, children=[
                    html.P("ALL CITIES", style={
                        'fontSize':'11px','fontWeight':'600','letterSpacing':'3px',
                        'opacity':'0.5','marginBottom':'6px',
                    }),
                    html.H2("Air Quality by City", style={
                        'fontSize':'26px','fontWeight':'600','letterSpacing':'-0.3px',
                    }),
                ]),
                html.Div(className='masonry', children=[
                    city_card(row) for _, row in city_avg.iterrows()
                ]),
            ]),

            # Charts
            html.Div(style={'marginBottom':'24px'}, children=[
                html.P("ANALYTICS", style={
                    'fontSize':'11px','fontWeight':'600','letterSpacing':'3px',
                    'opacity':'0.5','marginBottom':'6px',
                }),
                html.H2("Data Insights", style={
                    'fontSize':'26px','fontWeight':'600','letterSpacing':'-0.3px',
                }),
            ]),

            html.Div(style={
                'display':'grid',
                'gridTemplateColumns':'1fr 1fr',
                'gap':'20px','marginBottom':'20px',
            }, children=[
                html.Div(id='chart-card-1', className='glass', style={
                    'borderRadius':'24px','padding':'28px',
                    'transition':'background 0.8s ease, border-color 0.8s ease',
                }, children=[
                    html.P("AQI DISTRIBUTION", style={'fontSize':'10px','fontWeight':'600','letterSpacing':'2.5px','opacity':'0.5','marginBottom':'4px'}),
                    html.H3("How AQI Spreads", style={'fontSize':'17px','fontWeight':'600','marginBottom':'16px'}),
                    dcc.Graph(id='fig1', config={'displayModeBar':False}, style={'height':'300px'}),
                ]),
                html.Div(id='chart-card-2', className='glass', style={
                    'borderRadius':'24px','padding':'28px',
                    'transition':'background 0.8s ease, border-color 0.8s ease',
                }, children=[
                    html.P("CITY COMPARISON", style={'fontSize':'10px','fontWeight':'600','letterSpacing':'2.5px','opacity':'0.5','marginBottom':'4px'}),
                    html.H3("Average AQI per City", style={'fontSize':'17px','fontWeight':'600','marginBottom':'16px'}),
                    dcc.Graph(id='fig2', config={'displayModeBar':False}, style={'height':'300px'}),
                ]),
            ]),

            html.Div(style={
                'display':'grid',
                'gridTemplateColumns':'1fr 1fr',
                'gap':'20px','marginBottom':'20px',
            }, children=[
                html.Div(id='chart-card-3', className='glass', style={
                    'borderRadius':'24px','padding':'28px',
                    'transition':'background 0.8s ease, border-color 0.8s ease',
                }, children=[
                    html.P("SEASONAL PATTERN", style={'fontSize':'10px','fontWeight':'600','letterSpacing':'2.5px','opacity':'0.5','marginBottom':'4px'}),
                    html.H3("Monthly AQI Trend", style={'fontSize':'17px','fontWeight':'600','marginBottom':'16px'}),
                    dcc.Graph(id='fig3', config={'displayModeBar':False}, style={'height':'300px'}),
                ]),
                html.Div(id='chart-card-4', className='glass', style={
                    'borderRadius':'24px','padding':'28px',
                    'transition':'background 0.8s ease, border-color 0.8s ease',
                }, children=[
                    html.P("POLLUTANT CORRELATION", style={'fontSize':'10px','fontWeight':'600','letterSpacing':'2.5px','opacity':'0.5','marginBottom':'4px'}),
                    html.H3("What Drives AQI?", style={'fontSize':'17px','fontWeight':'600','marginBottom':'16px'}),
                    dcc.Graph(id='fig4', config={'displayModeBar':False}, style={'height':'300px'}),
                ]),
            ]),

            html.Div(style={
                'display':'grid',
                'gridTemplateColumns':'1fr 1fr',
                'gap':'20px',
            }, children=[
                html.Div(id='chart-card-5', className='glass', style={
                    'borderRadius':'24px','padding':'28px',
                    'transition':'background 0.8s ease, border-color 0.8s ease',
                }, children=[
                    html.P("POLLUTANT SCATTER", style={'fontSize':'10px','fontWeight':'600','letterSpacing':'2.5px','opacity':'0.5','marginBottom':'4px'}),
                    html.H3("PM2.5 vs AQI", style={'fontSize':'17px','fontWeight':'600','marginBottom':'16px'}),
                    dcc.Graph(id='fig5', config={'displayModeBar':False}, style={'height':'300px'}),
                ]),
                html.Div(id='chart-card-6', className='glass', style={
                    'borderRadius':'24px','padding':'28px',
                    'transition':'background 0.8s ease, border-color 0.8s ease',
                }, children=[
                    html.P("YEARLY TREND", style={'fontSize':'10px','fontWeight':'600','letterSpacing':'2.5px','opacity':'0.5','marginBottom':'4px'}),
                    html.H3("Top 5 Cities Over Time", style={'fontSize':'17px','fontWeight':'600','marginBottom':'16px'}),
                    dcc.Graph(id='fig6', config={'displayModeBar':False}, style={'height':'300px'}),
                ]),
            ]),

        ]),

        # Footer
        html.Footer(style={
            'textAlign':'center','padding':'32px',
            'fontSize':'12px','opacity':'0.4','letterSpacing':'1px',
            'position':'relative','zIndex':'1',
        }, children="Breath of the City · Air Quality Index Analysis · India 2015–2020"),

    ]),
])


# ── Callbacks ───────────────────────────────────────────────────
@app.callback(
    Output('mode-store',   'data'),
    Output('toggle-btn',   'children'),
    Input('toggle-btn',    'n_clicks'),
    prevent_initial_call=True,
)
def toggle_mode(n):
    if n % 2 == 1:
        return 'polluted', '☁  Polluted Mode'
    return 'clean', '◎  Clean Air Mode'


@app.callback(
    Output('page-wrapper',  'style'),
    Output('navbar',        'style'),
    Output('hero',          'style'),
    Output('navbar',        'children'),
    Output('chart-card-1',  'style'),
    Output('chart-card-2',  'style'),
    Output('chart-card-3',  'style'),
    Output('chart-card-4',  'style'),
    Output('chart-card-5',  'style'),
    Output('chart-card-6',  'style'),
    Output('fig1', 'figure'),
    Output('fig2', 'figure'),
    Output('fig3', 'figure'),
    Output('fig4', 'figure'),
    Output('fig5', 'figure'),
    Output('fig6', 'figure'),
    Input('mode-store', 'data'),
)
def apply_theme(mode):

    if mode == 'clean':
        bg_page   = 'linear-gradient(160deg, #e8f5f0 0%, #f0f4ff 40%, #fef9f0 100%)'
        nav_bg    = 'rgba(255,255,255,0.65)'
        nav_bdr   = '1px solid rgba(0,0,0,0.08)'
        nav_color = '#1a1a2e'
        hero_bg   = 'linear-gradient(160deg,#c8f0e0 0%,#dce8ff 50%,#fde8c8 100%)'
        chart_bg  = 'rgba(255,255,255,0.55)'
        chart_bdr = '1px solid rgba(255,255,255,0.8)'
        plot_bg   = 'rgba(255,255,255,0)'
        plot_grid = 'rgba(0,0,0,0.06)'
        text_c    = '#1a1a2e'
        sub_c     = '#6b7280'
        toggle_lbl = '◎  Clean Air Mode'
    else:
        bg_page   = 'linear-gradient(160deg,#1a0a2e 0%,#16213e 40%,#1a1a2e 100%)'
        nav_bg    = 'rgba(20,10,40,0.75)'
        nav_bdr   = '1px solid rgba(255,255,255,0.08)'
        nav_color = '#e2e8f0'
        hero_bg   = 'linear-gradient(160deg,#2d1b4e 0%,#1a2744 50%,#2d1a1a 100%)'
        chart_bg  = 'rgba(255,255,255,0.05)'
        chart_bdr = '1px solid rgba(255,255,255,0.1)'
        plot_bg   = 'rgba(0,0,0,0)'
        plot_grid = 'rgba(255,255,255,0.07)'
        text_c    = '#e2e8f0'
        sub_c     = '#94a3b8'
        toggle_lbl = '☁  Polluted Mode'

    page_style = {
        'minHeight':'100vh',
        'background': bg_page,
        'color': text_c,
        'transition':'background 0.8s ease, color 0.6s ease',
    }
    nav_style = {
        'position':'sticky','top':'0','zIndex':'100',
        'padding':'0 48px','height':'64px',
        'display':'flex','alignItems':'center','justifyContent':'space-between',
        'background': nav_bg,
        'borderBottom': nav_bdr,
        'color': nav_color,
        'transition':'background 0.8s ease',
        'backdropFilter':'blur(16px)',
    }
    nav_children = [
        html.Div(style={'display':'flex','alignItems':'center','gap':'10px','color':nav_color}, children=[
            html.Div('◎', style={'fontSize':'22px'}),
            html.Span("Breath of the City", style={'fontSize':'16px','fontWeight':'600','letterSpacing':'0.5px'}),
        ]),
        html.Div(style={'display':'flex','gap':'32px','alignItems':'center','color':nav_color}, children=[
            html.Span("Overview",   className='nav-link', style={'fontSize':'13px','fontWeight':'500','cursor':'pointer'}),
            html.Span("Cities",     className='nav-link', style={'fontSize':'13px','fontWeight':'500','cursor':'pointer'}),
            html.Span("Analytics",  className='nav-link', style={'fontSize':'13px','fontWeight':'500','cursor':'pointer'}),
            html.Div(id='toggle-btn', n_clicks=0 if mode=='clean' else 1, className='toggle-btn', style={
                'display':'flex','alignItems':'center','gap':'8px',
                'background':'rgba(128,128,128,0.2)',
                'border':f'1px solid {"rgba(0,0,0,0.15)" if mode=="clean" else "rgba(255,255,255,0.15)"}',
                'borderRadius':'20px','padding':'6px 14px',
                'fontSize':'12px','fontWeight':'500','color':nav_color,
            }, children=toggle_lbl),
        ]),
    ]
    hero_style = {
        'position':'relative','overflow':'hidden',
        'padding':'80px 48px 60px',
        'textAlign':'center','zIndex':'1',
        'background': hero_bg,
        'color': text_c,
        'transition':'background 0.8s ease',
    }
    card_style = {
        'borderRadius':'24px','padding':'28px',
        'background': chart_bg,
        'border': chart_bdr,
        'backdropFilter':'blur(16px)',
        'transition':'background 0.8s ease, border-color 0.8s ease',
        'color': text_c,
    }

    font = 'Poppins, sans-serif'
    f1,f2,f3,f4,f5,f6 = make_chart(plot_bg, chart_bg, text_c, sub_c, plot_grid, font)

    return (page_style, nav_style, hero_style, nav_children,
            card_style, card_style, card_style, card_style, card_style, card_style,
            f1, f2, f3, f4, f5, f6)


if __name__ == '__main__':
    app.run(debug=True)