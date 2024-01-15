def normal(array):
    
    return np.abs( array ) / np.max( np.abs( array ) )

def diagPlot_cuts(theta, hor, vert, cutTitle):
    
    theta = theta * 180/pi
    hor = 2*linTodB( np.abs(hor)/np.max(np.abs(hor)) )
    vert = 2*linTodB( np.abs(vert) / np.max(np.abs(vert)) )
    
    plt.figure()
    plt.plot(theta, hor, theta, vert)
    plt.xlabel('Theta (deg)')
    plt.ylabel('Amplitude (dB)')
    plt.title(cutTitle)
    plt.legend(['Phi=0 Cut', 'Phi=pi/2 Cut'])
    plt.grid()
    
    
def diagPlot_uv(u, v, patt, title):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.pcolormesh(u, v, np.abs(patt) )
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    ax.set_title(title)
    ax.grid()
    
    t = np.linspace(0, 2 * pi, 500)
    ax.plot(np.cos(t), np.sin(t), linewidth=1)
    ax.set_aspect('equal', 'box')
    
    
def t1():
    #Test basic rectangular arrays
    
    #Test 4x4 dx=dy=0.5
    title = '4x4, 0.5 0.5, w=1'        
    elements = Array_Config(4, 4, 0.5, 0.5).rectArray()
    plt.figure()
    plt.scatter(elements['x'], elements['y'])
    plt.title(title)
    plt.grid()
    
    array1 = Array_2D(elements)
    
    theta = np.linspace(-pi/2, pi/2, 2000)
    phi = 0
    hor = array1.arrayFactor(theta, phi)
    phi = pi/2
    vert = array1.arrayFactor(theta, phi)
    
    diagPlot_cuts(theta, hor, vert, title)
    
    u = np.linspace(-2, 2, 500)
    v = np.linspace(-2, 2, 500)
    
    uu, vv = np.meshgrid(u, v)
    
    uvPatt = array1.uniPattUV(uu, vv)
    
    diagPlot_uv(u, v, uvPatt, title)

    
    #Test 4x4 dx=dy=1  
    title = '4x4 dx = 0.5, dy = 0.75'
    elements = Array_Config(4, 4, 1, 1).rectArray()
    plt.figure()
    plt.scatter(elements['x'], elements['y'])
    plt.title(title)
    plt.grid()
    
    array1 = Array_2D(elements)
    
    theta = np.linspace(-pi/2, pi/2, 2000)
    phi = 0
    hor = array1.arrayFactor(theta, phi)
    phi = pi/2
    vert = array1.arrayFactor(theta, phi)
    
    diagPlot_cuts(theta, hor, vert, title)
    
    u = np.linspace(-2, 2, 500)
    v = np.linspace(-2, 2, 500)
    
    uu, vv = np.meshgrid(u, v)
    
    uvPatt = array1.uniPattUV(uu, vv)
    
    diagPlot_uv(u, v, uvPatt, title)
    
    #Test 4x4 dx = 0.5, dy = 0.75
    title = '4x4 dx = 0.5, dy = 0.75'
    elements = Array_Config(4, 4, 0.5, 0.75).rectArray()
    plt.figure()
    plt.scatter(elements['x'], elements['y'])
    plt.title(title)
    plt.grid()
    
    array1 = Array_2D(elements)
    
    theta = np.linspace(-pi/2, pi/2, 2000)
    phi = 0
    hor = array1.arrayFactor(theta, phi)
    phi = pi/2
    vert = array1.arrayFactor(theta, phi)
    
    diagPlot_cuts(theta, hor, vert, title)
    
    u = np.linspace(-2, 2, 500)
    v = np.linspace(-2, 2, 500)
    
    uu, vv = np.meshgrid(u, v)
    
    uvPatt = array1.uniPattUV(uu, vv)
    
    diagPlot_uv(u, v, uvPatt, title)
    
    
    
    
if __name__ == '__main__':
    
    a1 = Array_Config(15, 15, 0.99, 0.99)
    
    elements = a1.triangArray()
    
    plt.figure()
    plt.scatter(elements['x'], elements['y'])
    plt.grid()
    
    ap1 = Array_2D(elements)
    u = np.linspace(-2, 2, 500)
    v = u
    uu, vv = np.meshgrid(u, v)
    patt = ap1.uniPattUV(uu, vv)
    diagPlot_uv(u, v, patt, 'Normal One')
    
    theta = np.linspace(-pi/2, pi/2, 1000)
    phi = 0
    hor = ap1.arrayFactor(theta, phi)
    vert = ap1.arrayFactor(theta, pi/2)
    diagPlot_cuts(theta, hor, vert, 'Normal')
    plt.ylim([-30, 0])
    
    
    # posList = [(3.5, 3), (3, 2.5), (4, 2.5), (4, 3.5), (3, 3.5), (3.5, 2), (3.5, 4), (2.5, 3), (4.5, 3)]
    # for pos in posList:
    #     elements = a1.delElement(pos[0], pos[1], elements)
    
    elements = a1.delElements( (2.5, 4.5), (2.5, 4.5), elements)
    
    plt.figure()
    plt.scatter(elements['x'], elements['y'])
    plt.grid()
    
    ap1 = Array_2D(elements)
    u = np.linspace(-2, 2, 500)
    v = u
    uu, vv = np.meshgrid(u, v)
    patt = ap1.uniPattUV(uu, vv)
    diagPlot_uv(u, v, patt, 'Holey One')

    theta = np.linspace(-pi/2, pi/2, 1000)
    phi = 0
    hor = ap1.arrayFactor(theta, phi)
    vert = ap1.arrayFactor(theta, pi/2)
    diagPlot_cuts(theta, hor, vert, 'Holey') 
    plt.ylim([-30, 0])
    
    hor_normal = 2*linTodB( normal( hor ) )
    
    
    

        
    # peaks = find_peaks(hor_normal)[0]
    
    
    # print(len(peaks))
    # cntrPeakIdx = int( len(peaks)/2 )
    # mainLobe = peaks[ cntrPeakIdx ]
    # print(mainLobe)
    # sideLobes = peaks[ [cntrPeakIdx - 1, cntrPeakIdx + 1] ]
    # print(sideLobes)
    # print( hor_normal[peaks] )
    # print( hor_normal[mainLobe] )
    # print( hor_normal[sideLobes] )
    
    print( getSLL(hor_normal) )
    
    
    
    
    
    # t1()