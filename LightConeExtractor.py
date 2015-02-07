from __future__ import division
import numpy as np

def get_p(past):
    indices = [i // 2 for i in past.shape[1:-1]]
    pslice = past[0]
    for i in indices:
        pslice = pslice[i]
    return pslice


class LightConeExtractor(object):
    def __init__(self, speed=1, h_p = 2):
        self.c = speed
        self.h_p = h_p

    def extract(self, data):
        """ Goes through data and for each point extracts 
            past light cone and future light cone point, 
            adding them to a list. NOTE: each light cone
            is flattened to a one-dimensional array, with
            the future point at index -1 (last element).
            input: data (numpy matrix or array)
            Assumes time dimension is first dimension.
        """        
        if data.ndim == 1:
            return self.extract_0D(data)
        elif data.ndim == 2:
            return self.extract_1D(data)
        elif data.ndim == 3:
            return self.extract_2D(data)
        else:
            print "Currently there is no support for data of dimensionality (%d + 1)D." % (data.ndim - 1)
    def extract_0D(self, data):
        """ Time series data """
        coordinates = [i for i in xrange(0, len(data) - self.h_p)]
        return np.array([data[i:i+self.h_p] for i in xrange(0, len(data) - self.h_p)]), np.array(coordinates)

    def extract_1D(self, data):
        """ One dimensional over time (like 1D cellular automata) """
        light_cones = []
        timesteps = data.shape[0]
        ns = [1,]
        coordinates = []
        for i in range(self.h_p):
            ns.append(2 * self.c + ns[-1])
        shim_width = self.c * self.h_p
        for t in xrange(self.h_p, timesteps):            
            for i in xrange(shim_width, data.shape[1] - shim_width):                
                ''' Get FLC point '''
                lc = [data[t,i]]
                ''' Get rest of pyramid '''
                for k in range(1, self.h_p + 1):
                    r = ns[k] // 2
                    patch = data[t-k, i-r:i+r+1]
                    lc.extend(patch.ravel())
                light_cones.append(lc[::-1])
                coordinates.append((t,i))
        lc_array = np.array(light_cones)
        return lc_array.reshape((lc_array.shape[0], lc_array.shape[1], 1)), np.array(coordinates)

    def extract_2D(self, data):
        """ Matrices over time (tensors, with time as initial dimension).
            (The matrices are stacked flat in rows).
        """
        light_cones = []
        timesteps = data.shape[0]
        ns = [1,]
        elements_per_level = []
        coordinates = []
        for i in range(self.h_p):
            ns.append(2 * self.c + ns[-1])
            elements_per_level.append(ns[-1]**2)
        shim_width = self.c * self.h_p
        for t in xrange(self.h_p, timesteps):            
            for i in xrange(shim_width, data.shape[1] - shim_width):
                for j in xrange(shim_width, data.shape[2] - shim_width):
                    ''' Get FLC point '''
                    lc = [data[t,i,j]]
                    ''' Get rest of pyramid '''
                    for k in range(1, self.h_p + 1):
                        r = ns[k] // 2
                        patch = data[t-k, i-r:i+r+1, j-r:j+r+1]
                        lc.extend(patch.ravel())
                    light_cones.append(lc[::-1])
                    coordinates.append((t,i,j))
        lc_array = np.array(light_cones)
        return lc_array.reshape((lc_array.shape[0], lc_array.shape[1], 1)), np.array(coordinates)

if __name__ == "__main__":
    x = np.random.random(size=(10,256,256))
    l = LightConeExtractor(speed=1, h_p=1)
    lcs, coords = l.extract(x)
    print lcs.shape
    print coords
    
