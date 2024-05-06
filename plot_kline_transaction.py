import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection,PolyCollection

def draw_transaction(product_name):
    import os

    # Get current working directory
    current_directory = os.getcwd()
    # Read data from CSV file
    data = pd.read_csv( current_directory+'/Data/'+product_name+'/source.csv').iloc[:,0:5].iloc[::-1]
    # Create a new column 'trade_date' with a range of numbers
    data['trade_date'] = range(0, len(data))
    # Select specific columns from the data
    df = data.loc[:,['trade_date','OPEN','HIGH','LOW','CLOSE','VOLUME']]

    # Prepare data for plotting
    date_tickers=df.trade_date.values
    matix = df.values
    xdates = matix[:, 0]

    # Configure plot styles
    plt.rc('figure', facecolor='white')  # Set figure background to white
    plt.rc('text', c='#f00000')
    plt.rc('axes', axisbelow=True, xmargin=0, fc='w', ec='#800000', lw=2, labelcolor='#800000', unicode_minus=False)
    plt.rc('xtick', c='#f43221')
    plt.rc('ytick', c='#f43221')
    plt.rc('grid', c='#f00000',  ls=':', lw=0.9)
    plt.rc('lines', lw=0.9)

    # Create figure and axes
    fig = plt.figure(figsize=(16,10),dpi=200)
    left, width = 0.06, 0.9
    ax1 = fig.add_axes([left, 0.5, width, 0.35])
    ax2 = fig.add_axes([left, 0.34, width, 0.15], sharex=ax1)
    ax3 = fig.add_axes([left, 0.13, width, 0.2], sharex=ax1)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)

    # Read log file and add patches to ax1 based on the log content
    f = open('Results/log/best_log_' + product_name + '.txt', 'r')
    lines = f.readlines()
    days = 1
    for line in lines:
        if 'Successfully Order' in line:
            ax1.add_patch(
                plt.Rectangle(
                    (days - 0.04, data['LOW'].min()),
                    0.8,
                    data['HIGH'].max() - data['LOW'].min(),
                    color='#FFA500',
                    alpha=1
                )
            )
        if 'Successfully Sell' in line:
            ax1.add_patch(
                plt.Rectangle(
                    (days - 0.04, data['LOW'].min()),
                    0.8,  # width长
                    data['HIGH'].max() - data['LOW'].min(),
                    color='#008000',
                    alpha=1
                )
            )
        days += 1

    def format_date(x, pos=None):
        return '' if x<0 or x>len(date_tickers)-1 else date_tickers[int(x)]

    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(max(int(len(df)/15), 5)))
    opens, closes, highs, lows = matix[:, 1], matix[:, 2], matix[:, 3], matix[:, 4]
    avg_dist_between_points = (xdates[-1] - xdates[0]) / float(len(xdates))
    delta = avg_dist_between_points / 4.0
    barVerts = [((date - delta, open), (date - delta, close), (date + delta, close), (date + delta, open)) for date, open, close in zip(xdates, opens, closes) ]
    rangeSegLow   = [ ((date, low), (date, min(open, close))) for date, low, open, close in zip(xdates, lows, opens, closes) ]
    rangeSegHigh  = [ ((date, high), (date, max(open, close))) for date, high, open, close in zip(xdates, highs, opens, closes) ]
    rangeSegments = rangeSegLow + rangeSegHigh
    cmap = {True: mcolors.to_rgba('#DC143C', 1.0), False: mcolors.to_rgba('#DC143C', 1.0)}
    inner_colors = [ cmap[opn < cls] for opn, cls in zip(opens, closes) ]
    cmap = {True: mcolors.to_rgba('#DC143C', 1.0), False: mcolors.to_rgba('#DC143C', 1.0)}
    updown_colors = [ cmap[opn < cls] for opn, cls in zip(opens, closes) ]
    ax1.add_collection(LineCollection(rangeSegments, colors=updown_colors, linewidths=0.7, antialiaseds=False))
    ax1.add_collection(PolyCollection(barVerts, facecolors=inner_colors, edgecolors=updown_colors, antialiaseds=False, linewidths=0.1))

    ax1.plot(xdates, pd.read_csv('Data/'+product_name+'/source.csv')['QPL+'].iloc[::-1], label='QPL+', color='#FF00FF')
    ax1.plot(xdates, pd.read_csv('Data/'+product_name+'/source.csv')['QPL-'].iloc[::-1], label='QPL-', color='#00FFFF')

    mav_colors = ['#d4ff07','#33ff81']
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
    ax1.set_ylabel('Price',color='#FF4500')


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
    ax2.set_ylabel('Volume',color='#FF4500')

    ax3.plot(pd.read_csv('Data/AUDUSD/source.csv')['RSI'].tolist(),label='RSI',linewidth=1.5)
    ax3.yaxis.set_ticks_position('left')
    ax3.legend(loc='lower right')
    ax3.grid(True)
    ax3.set_ylim([0,100])
    ax3.set_ylabel('RSI',color='#FF4500')
    ax3.set_xlabel('Trade Days',color='#FF4500')

    plt.savefig('Results/graph/Kline_transaction/'+product_name+'_KLine.png',dpi=400)
