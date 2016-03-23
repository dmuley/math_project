import numpy as np;
from scipy.interpolate import splrep;
from scipy.stats import linregress;
import matplotlib.pyplot as plt;
from mpl_toolkits.mplot3d import Axes3D;

xList = [];
yList = [];

SET = 1;

def onclick(event):
    button=event.button
    x=event.xdata
    y=event.ydata

    if button==1: plt.plot(x,y,'ro')
    if button!=1: plt.plot(x,y,'bo')
    #print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(event.button, event.x, event.y, event.xdata, event.ydata)

    xList.append(event.xdata);
    yList.append(event.ydata);
    
def displayPlot(pic = 'Picture1.png'):
	im = plt.imread(pic)
	fig, ax=plt.subplots()
	ax.imshow(im)
	ax.autoscale(False)
	cid = fig.canvas.mpl_connect('button_press_event', onclick)

	plt.show();
	
	#do not consider the last 2 points, these are used for basis
	x = np.array(xList[:len(xList)-2])
	y = np.array(yList[:len(xList)-2])[np.argsort(xList[:len(xList)-2])]
	x = np.sort(x)
	
	xl = np.array([xList[len(xList) - 2], xList[len(xList) - 1]]);
	yl = np.array([yList[len(xList) - 2], yList[len(xList) - 1]]);

	return x, y, xl, yl
	
def computeRegression(x, y, xl, yl):
	x1 = np.linspace(x.min(), x.max(), 1001);
	h = np.polyfit(x, y, 7);
	
	o = linregress(xl, yl);
	print [o[1], o[0]];
	
	h2 = np.concatenate((np.zeros(6), [o[0], o[1]]))
	m = np.polyval(h - h2, x1);
	m2 = np.polyval(h - h2, x);
	
	print h - h2;
	print h2;
	
	plt.plot(x1, -m);
	plt.plot(x, -y + np.polyval(h2, x),".");
	plt.show();
	
	return h - h2, x1, -m
		
if SET == 0:
	xList = [];
	yList = [];	
	a, b, c, d = displayPlot('Picture1.png');
	xList = [];
	yList = [];
	e, f, g, h = displayPlot('Picture2.png');
	print "Polynomial 1: "
	o, p, q = computeRegression(a, b, c, d);
	print "Polynomial 2:"
	r, s, t = computeRegression(e, f, g, h);
	
	print "Lower and upper bound: "
	print np.sqrt(p * s)[-1];
	print np.sqrt(p * s)[0];

	scale = np.sqrt((p[-1]-p[0]) * (s[-1] - s[0]))/1001.;
	total_volume = q * t;
		
	coord = np.array([np.linspace(np.sqrt(p[0]*s[0]),np.sqrt(s[-1]*p[-1]), 1001), q, np.zeros(1001)]);
	coord2= np.array([np.linspace(np.sqrt(p[0]*s[0]),np.sqrt(s[-1]*p[-1]), 1001), np.zeros(1001), t]);
	
	fig = plt.figure();
	ax = fig.add_subplot(111, projection='3d');
	
	for n in np.linspace(0,np.pi * 2,100):
		coord5 = np.array([np.linspace(np.sqrt(p[0]*s[0]),np.sqrt(s[-1]*p[-1]), 1001), q * np.sin(n), t * np.cos(n)]);	
		ax.plot(xs=coord5[0], ys=coord5[1], zs=coord5[2]);
		
	ax.plot(xs=coord[0], ys=coord[1], zs=coord[2]);
	ax.plot(xs=coord2[0], ys=coord2[1], zs=coord2[2]);
		
	fig.show();

	total_volume *= scale / (25.1931**3);
	vol = np.sum(total_volume) * np.pi * 2.54**3;
	print vol

if SET == 1:
	xList = [];
	yList = [];	
	a, b, c, d = displayPlot('arm1.png');
	xList = [];
	yList = [];	
	e, f, g, h = displayPlot('arm1.png');
	
	print "Polynomial 1: "
	o, p, q = computeRegression(a, b, c, d);
	print "Polynomial 2: "
	r, s, t = computeRegression(e, f, g, h);
	
	print "Lower and upper bound: "
	print np.sqrt(p * s)[-1];
	print np.sqrt(p * s)[0];

	
	scale = np.sqrt((p[-1]-p[0]) * (s[-1] - s[0]))/1001.;
	total_volume = (q - t)**2	
	
	coord = np.array([np.linspace(np.sqrt(p[0]*s[0]),np.sqrt(s[-1]*p[-1]), 1001), q, np.zeros(1001)]);
	coord2= np.array([np.linspace(np.sqrt(p[0]*s[0]),np.sqrt(s[-1]*p[-1]), 1001), t, np.zeros(1001)]);
	coord3= np.array([np.linspace(np.sqrt(p[0]*s[0]),np.sqrt(s[-1]*p[-1]), 1001), (q + t)/2, (q - t)/2]);
	coord4= np.array([np.linspace(np.sqrt(p[0]*s[0]),np.sqrt(s[-1]*p[-1]), 1001), (q + t)/2, -(q - t)/2]);
	
	fig = plt.figure();
	ax = fig.add_subplot(111, projection='3d');
	
	for n in np.linspace(0,np.pi * 2,100):
		coord5 = np.array([np.linspace(np.sqrt(p[0]*s[0]),np.sqrt(s[-1]*p[-1]), 1001), (q + t)/2 + np.sin(n) * (q - t)/2, np.cos(n) * (q - t)/2]);	
		ax.plot(xs=coord5[0], ys=coord5[1], zs=coord5[2]);
		
	ax.plot(xs=coord[0], ys=coord[1], zs=coord[2]);
	ax.plot(xs=coord2[0], ys=coord2[1], zs=coord2[2]);
	ax.plot(xs=coord3[0], ys=coord3[1], zs=coord3[2]);
	ax.plot(xs=coord4[0], ys=coord4[1], zs=coord4[2]);
			
	total_volume *= scale / 25.1953**3 / 4.;
	vol = np.sum(total_volume) * np.pi * 2.54**3;
	print vol
