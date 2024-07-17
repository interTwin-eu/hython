import xarray as xr
import numpy as np

import matplotlib.pyplot as plt
from shapely.geometry import Point,Polygon
from functools import partial
from shapely.ops import transform,unary_union
import pyproj
from hython.utils import read_from_zarr

import shapely


EPSG_4326 = '+proj=longlat +datum=WGS84 +no_defs'
EPSG_3035 ='+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=WGS84 +units=m +no_defs'



def get_upstream_area(x,y,ldd):
    """
        Get LISFLOOD upstream catchment. 
        
        
    Args:
        x (int): Projected coordinates
        y (int): Projected coordinates

    Returns:
        dict (string:list): A dictionary with 'lats' and 'lons' keys and coordinates values. It can be converted to 
            a shape with ct.shapes.polygon or ct.shapes.multipolygon depending on the number of 
    
    """

    FILTER = np.array([[3,2,1],[6,5,4],[9,8,7]])
    DELTAX = np.array([[-1,0,1],
                      [-1,0,1],
                      [-1,0,1]])
    DELTAY = np.array([[-1,-1,-1],
                      [0,0,0],
                      [1,1,1]])

    nodata = np.nan #ldd.attrs['nodatavals'][0]
    #ldd = ldd.rename({"x":"lon","y":"lat"})


    ldd_sub = ldd#.drop(["band"],inplace=True).squeeze()


    ldd_sub = ldd_sub.where(ldd_sub != nodata,np.nan)


    start = (int(np.argwhere(ldd_sub.lon.values==x)[0]),
             int(np.argwhere(ldd_sub.lat.values==y)[0]))

    coords = [start]

    res = np.full(ldd_sub.shape,False)

    res[start[1],start[0]] = True

    
    while len(coords) >0:

        directions = []
        deltaxys = []
        moveto  = []
        for pointer,_ in enumerate(coords):
            
            coord = coords[pointer]
            
            try:
                # if starting point is at edge do something and the same if last point is at edge
                direction = ldd_sub.values[(coord[1]-1):(coord[1]+2),(coord[0]-1):(coord[0] + 2)] == FILTER
                res[(coord[1]-1):(coord[1]+2),(coord[0]-1):(coord[0] + 2)] =  (res[(coord[1]-1):(coord[1]+2),(coord[0]-1):(coord[0] + 2)]) | (direction)
            except:
                break
                coords = []
            directions.append(direction)


            deltaxy = np.vstack([DELTAX[direction],DELTAY[direction]])
            deltaxys.append(deltaxy)
            import pdb;pdb.set_trace()
            for istep in deltaxys:
                if istep.size <1:
                    pass
                else:
                    for i in range(istep.shape[1]):
                        moveto.append(istep[:,i] + coord)
            deltaxys.pop()

        coords = moveto


    da = xr.DataArray(data=res,coords=[ldd_sub.lat,ldd_sub.lon], dims=['lat','lon'])


    result = ldd_sub.where(da,drop=True)

    
    polygons = create_polygons(ds=result,pixelRes=5000)

    diss = unary_union(polygons)
    
    lons,lats = resolve_multipolygon(diss)


    ret = reproject(x=lons,y=lats,s_srs=EPSG_3035,t_srs=EPSG_4326)

    return ret,result



def resolve_multipolygon(data):
    if isinstance(data,shapely.geometry.multipolygon.MultiPolygon):
        lats,lons = [],[]
        for ipol in data:
            #print(ipol)
            lon,lat = ipol.exterior.coords.xy
            lats.append(lat.tolist())
            lons.append(lon.tolist())
    else:
        coords_out = data.exterior.coords.xy
        lons =  coords_out[0].tolist()
        lats =  coords_out[1].tolist()
        
        
    return lons,lats


def create_polygons(ds,pixelRes):

    polygons = []
    count = 1
    for x in ds.lon:
        for y in ds.lat:
            
            if np.isnan(ds.sel(lat =y,lon=x,method="nearest").values):
                pass
            else:
                polygons.append(
                    Polygon([(x - pixelRes / 2, y + pixelRes / 2), (x + pixelRes / 2, y + pixelRes / 2), (x + pixelRes / 2, y - pixelRes / 2),
                         (x - pixelRes / 2, y - pixelRes / 2)]))
                count += 1

    return polygons

def reproject(x,y,s_srs=EPSG_4326,t_srs=EPSG_3035):

    # multi 
    
    #print(x)
    #print(y)
    ismulti =False     
    if any(isinstance(i, list) for i in x):
        ismulti = True
        

    #pyproj.__version__
    if not isinstance(x,list) or not isinstance(y,list):
        x = [x]
        y = [y]

    s_srs_proj = pyproj.Proj(s_srs)
    t_srs_proj = pyproj.Proj(t_srs)

    project = partial(
      pyproj.transform,
      s_srs_proj , # source coordinate system
      t_srs_proj)

    if ismulti: 
        lats,lons = [],[]
        for _x,_y in zip(x,y): # multipolygon
            innerlats,innerlons = [],[]
            for x2,y2 in zip(_x,_y): # polygon
                geom = Point(x2,y2)
                lon,lat = transform(project,geom).x,transform(project,geom).y
                innerlats.append(lat)
                innerlons.append(lon)
            # each polygon needs to have the same start and end coordinates
            innerlats.extend([innerlats[0]])
            innerlons.extend([innerlons[0]])
            lats.append(innerlats)
            lons.append(innerlons)
 
    else:                
        geoms = []
        for _x,_y in zip(x,y):
            geoms.append(Point(_x,_y))
    
        lats,lons = [],[]
        for geom in geoms:
            lon,lat = transform(project,geom).x,transform(project,geom).y
            lats.append(lat)
            lons.append(lon)

    #if t_srs_proj.crs.is_projected:
    #    lats = list(map(int,lats))
    #    lons = list(map(int,lons))

    return {'lats':lats,'lons':lons}



def plot_upstream(df,ups_area,catch,labels=True):
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111,projection=crs.LambertAzimuthalEqualArea())
    #ax.coastlines()
    #ax.gridlines(draw_labels=True)
    s1 = ups_area.plot.pcolormesh(ax=ax,add_colorbar=False);
    #s1 = ldd_sub.plot.pcolormesh(ax=ax,add_colorbar=False);
    ax.add_geometries(catch["geometry"],crs=crs.LambertAzimuthalEqualArea(),facecolor=None,
                      edgecolor='red',alpha=0.4,linewidth=1.5)
    
    ax.add_geometries(df["geometry"],crs=crs.LambertAzimuthalEqualArea(),facecolor=None,
                      edgecolor='white',alpha=0.4,linewidth=1.5)
    
    plt.colorbar(s1,ax=ax)
    #catch.plot(ax=ax,color="green")
    ax.scatter(x, y, transform=crs.LambertAzimuthalEqualArea(),c="red",s=100)
    
    if labels:
        texts =[]
        for x1, y1, label in zip(sub.center.x, sub.center.y, sub["ldd"]):
            texts.append(ax.text(x1, y1, label, fontsize = 16,color="white"))
        
    lon = int(ups_area.lon.min())
    ticks = range(lon,(lon+ups_area.shape[1]*5000),5000)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(i) for i,_ in enumerate(ticks)])
    
    lat = int(ups_area.lat.min())
    ticks = range(lat,(lat+ups_area.shape[0]*5000),5000)
    ax.set_yticks(ticks)
    
    labely = [str(i) for i,_ in enumerate(ticks)]
    labely.reverse()
    ax.set_yticklabels(labely)
    
    plt.show()


def sel(x, coords):
    diff = np.inf
    val = 0
    for i in coords:    
        if np.abs(x - i) < diff:
            diff = np.abs(x - i)
            val = i
            
    return val

if __name__ == "__main__":


    

    
    
    #ldd = xr.open_rasterio('../data/ldd.tif')

    SURROGATE_INPUT = "https://eurac-eo.s3.amazonaws.com/INTERTWIN/SURROGATE_INPUT/adg1km_eobs_original.zarr/"
    ldd = read_from_zarr(SURROGATE_INPUT, group="xs").wflow_ldd

    x,y = sel(11, ldd.lon.values), sel(45.8, ldd.lat.values) #4382500,2437500
    
    # uparea = xr.open_rasterio('../data/uparea.tif')
    # df = gpd.read_file('../data/ec_upArea.shp')
    
    
    # catch = gpd.read_file('../data/po.shp')
    
    
    # nodata = ldd.attrs['nodatavals'][0]


    # ldd = ldd.rename({"x":"lon","y":"lat"})
    # uparea = uparea.rename({"x":"lon","y":"lat"})
    
    # ldd = ldd.where(ldd != nodata,np.nan)
    # uparea = uparea.where(uparea != nodata,np.nan)
    
    
    ret,res = get_upstream_area(x,y,ldd)
    
    res.plot()
    
    plt.show()
    
    #import pdb;pdb.set_trace()
    # bbox = catch.unary_union.envelope.bounds
    # factor = 10000
    
    
    # ldd_sub = ldd.where((ldd.lat>=bbox[1]-factor) & (ldd.lat <=bbox[3]+factor) & (ldd.lon>=bbox[0]- factor) & (ldd.lon <=bbox[2] + factor),drop=True)
    # uparea_sub = uparea.where((uparea.lat>=bbox[1]-factor) & (uparea.lat <=bbox[3]+factor) & (uparea.lon>=bbox[0]- factor) & (uparea.lon <=bbox[2] + factor),drop=True)
    
    # uparea_sub = uparea_sub.drop(["band"],inplace=True).squeeze()
    
    # ldd_sub = ldd_sub.drop(["band"],inplace=True).squeeze()
    
    # df["center"] = df["geometry"].centroid
    # print(ret)
    
    
    print(ret)