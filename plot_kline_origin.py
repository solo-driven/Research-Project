import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection, PolyCollection

def draw_origin(product_name):
    data = pd.read_csv('Data/'+product_name+'/source.csv').iloc[:,0:5].iloc[::-1]
    data['trade_date'] = range(0, len(data))
    df = data.loc[:,['trade_date','OPEN','HIGH','LOW','CLOSE','VOLUME']]

    date_tickers = df.trade_date.values
    matix = df.values
    xdates = matix[:, 0]

    plt.rc('font', family='Microsoft YaHei')
    plt.rc('figure', facecolor='white', figsize=(16, 10), dpi=200)  # white background and figure size
    plt.rc('text', c='k')  # black text
    plt.rc('axes', axisbelow=True, xmargin=0, fc='g', ec='k', lw=2, labelcolor='k', unicode_minus=False)  # black axes
    plt.rc('xtick', c='k')  # black xticks
    plt.rc('ytick', c='k')  # black yticks
    plt.rc('grid', c='k', ls=':', lw=0.9)  # black grid
    plt.rc('lines', lw=0.9)

    fig = plt.figure()
    left, width = 0.06, 0.9
    ax1 = fig.add_axes([left, 0.5, width, 0.35])
    ax2 = fig.add_axes([left, 0.34, width, 0.15], sharex=ax1)
    ax3 = fig.add_axes([left, 0.13, width, 0.2], sharex=ax1)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)

    def format_date(x, pos=None):
        return '' if x<0 or x>len(date_tickers)-1 else date_tickers[int(x)]

    # Set the x-axis label formatter to the custom 'format_date' function
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
    # Set the x-axis major tick locations to be at intervals of the maximum of either 5 or the length of the dataframe divided by 15
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(max(int(len(df)/15), 5)))
    # Extract the open, close, high, and low values from the matrix
    opens, closes, highs, lows = matix[:, 1], matix[:, 2], matix[:, 3], matix[:, 4]
    # Calculate the average distance between points in the xdates array
    avg_dist_between_points = (xdates[-1] - xdates[0]) / float(len(xdates))
    # Calculate the delta, which is a quarter of the average distance between points
    delta = avg_dist_between_points / 4.0

    # Create the vertices for the bars of the plot
    barVerts = [((date - delta, open), (date - delta, close), (date + delta, close), (date + delta, open)) for date, open, close in zip(xdates, opens, closes) ]
    rangeSegLow = [ ((date, low), (date, min(open, close))) for date, low, open, close in zip(xdates, lows, opens, closes) ]
    rangeSegHigh = [ ((date, high), (date, max(open, close))) for date, high, open, close in zip(xdates, highs, opens, closes) ]
    
    # Combine the low and high range segments
    rangeSegments = rangeSegLow + rangeSegHigh
    # Create a color map for the plot
    cmap = {True: mcolors.to_rgba('#DC143C', 1.0), False: mcolors.to_rgba('#DC143C', 1.0)}
    # Determine the inner colors of the bars based on whether the open price is less than the close price
    inner_colors = [ cmap[opn < cls] for opn, cls in zip(opens, closes) ]
    cmap = {True: mcolors.to_rgba('#DC143C', 1.0), False: mcolors.to_rgba('#DC143C', 1.0)}

    # Determine the updown colors of the bars based on whether the open price is less than the close price
    updown_colors = [ cmap[opn < cls] for opn, cls in zip(opens, closes) ]
    # Add the range segments to the plot as a LineCollection
    ax1.add_collection(LineCollection(rangeSegments, colors=updown_colors, linewidths=0.7, antialiaseds=False))
    # Add the bars to the plot as a PolyCollection
    ax1.add_collection(PolyCollection(barVerts, facecolors=inner_colors, edgecolors=updown_colors, antialiaseds=False, linewidths=0.1))
    
    # Plot the 'QPL+' data from the source.csv file
    ax1.plot(xdates, pd.read_csv('Data/'+product_name+'/source.csv')['QPL+'].iloc[::-1], label='QPL+', color='#FFA500')

    # Plot the 'QPL-' data from the source.csv file
    ax1.plot(xdates, pd.read_csv('Data/'+product_name+'/source.csv')['QPL-'].iloc[::-1], label='QPL-', color='#ADD8E6')

    # Define the colors and periods for the moving averages
    mav_colors = ['#FFFF00', '#000000']
    mav_period = [5, 21]
    n = len(df)
    for i in range(len(mav_period)):
        if n >= mav_period[i]:
            mav_vals = df['CLOSE'].rolling(mav_period[i]).mean().values
            ax1.plot(xdates, mav_vals, c=mav_colors[i%len(mav_colors)], label='MA'+str(mav_period[i]))
    ax1.set_title(product_name)
    ax1.grid(True)
    ax1.legend(loc='upper right')
    ax1.xaxis_date()
    ax1.set_ylabel('Price', color='#FF4500')

    barVerts = [((date - delta, 0), (date - delta, vol), (date + delta, vol), (date + delta, 0)) for date, vol in zip(xdates, matix[:,5]) ]
    ax2.add_collection(PolyCollection(barVerts, facecolors=inner_colors, edgecolors=updown_colors, antialiaseds=False, linewidths=0.1)) 
    if n>=5:
        vol5 = df['VOLUME'].rolling(5).mean().values
        ax2.plot(xdates, vol5, c='y', label='VOL5')
    if n>=21:
        vol10 = df['VOLUME'].rolling(21).mean().values
        ax2.plot(xdates, vol10, c='g', label='VOL21')
    ax2.yaxis.set_ticks_position('left')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    ax2.set_ylabel('Volume', color='#FF4500')

    ax3.plot(pd.read_csv('Data/AUDUSD/source.csv')['RSI'].tolist(), label='RSI', linewidth=1.5, color='#00FF00')
    ax3.yaxis.set_ticks_position('left')
    ax3.legend(loc='lower right')
    ax3.grid(True)
    ax3.set_ylim([0,100])
    ax3.set_ylabel('RSI', color='#FF4500')
    ax3.set_xlabel('Trade Days', color='#FF4500')

    plt.savefig('Results/graph/Kline_origin/'+product_name+'_KLine.png', dpi=400)

products = ['AUDUSD','AIRBUS','GOOGLE','USD100M1','XAUUSD']

for product_name in products:
    draw_origin(product_name)
