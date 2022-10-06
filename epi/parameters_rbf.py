# Definition of the parameters of the models
# At the moment parameters depend on space and time
import numpy as np
from scipy.special import erfc
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
from lmfit import Model
import functools
import datetime

# Mask function
def maskParams(params,m_mask):
    m_mask = np.invert(m_mask)
    return(np.ma.compressed( np.ma.masked_array( params, mask=m_mask) ))

def expgaussian(x, amplitude=1, center=0, sigma=1.0, gamma=1.0):
    """ an alternative exponentially modified Gaussian."""
    dx = center-x
    return amplitude* np.exp(gamma*dx) * erfc( dx/(np.sqrt(2)*sigma))

def EMGextrapol(x,y):

    model =  Model(expgaussian)
    params = model.make_params(sigma=10, gamma=0.01, amplitude=y.max(), center=y.argmax())
    result  = model.fit(y, params, x=x,  nan_policy='propagate')

    return result

# Utility for reading a section
def readSection(content,section,values):
    counter = 0
    data = np.zeros(values)
    sec_content = []
    found = False

    if values == 0:
        return()

    # extract from section
    for line in content:
        if line.startswith(section) or found:
            found = True
            sec_content.append(line)
        else:
            pass

    for line in sec_content:
        if line.startswith(section):
            pass
        elif line.startswith(b'#'):
            pass
        elif not line:
            pass
        else:
            tokens = line.split()
            for v in tokens:
                data[counter] = float(v)
                counter = counter + 1
                if counter == values:
                    return data

    return

def readTimes(content,section,values):
    counter = 0
    data = []
    sec_content = []
    found = False

    if values == 0:
        return()

    # extract from section
    for line in content:
        if line.startswith(section) or found:
            found = True
            sec_content.append(line)
        else:
            pass

    for line in sec_content:
        if line.startswith(section):
            pass
        elif line.startswith(b'#'):
            pass
        elif not line:
            pass
        else:
            data.append(datetime.date.fromisoformat(line.decode("utf-8").replace('\n','')))  
            counter = counter + 1
            if counter == values:
                return data

    return


# Params class
class Params():

    def __init__(self,dataStart):
        self.nParams = 0
        self.nSites = 0
        self.nPhases = 0
        self.estimated = False
        self.times = np.zeros(0)
        self.dataStart = dataStart
        self.dataEnd = 0
        self.degree = 0
        self.RBF = []
        self.extrapolator = []
        self.scenario = np.zeros((0,2)) 
        self.constant = np.zeros(0)
        self.constantSites = np.zeros(0)
        self.params  = np.zeros((0, 0))
        self.params_time = np.zeros((0,0))
        self.mask    = np.zeros((0, 0))
        self.lower_bounds = np.zeros((0, 0))
        self.upper_bounds = np.zeros((0, 0))

    def get(self):
        return(self.params)

    def getMask(self):
        return( np.array(self.mask, dtype=bool) )

    def getConstant(self):
        return( np.array(self.constant, dtype=bool) )

    def getConstantSites(self):
        return( np.array(self.constantSites, dtype=bool) )

    def getLowerBounds(self):
        return(self.lower_bounds)

    def getUpperBounds(self):
        return(self.upper_bounds)

    def save(self,paramFileName):
        if paramFileName.lower().endswith(('.csv', '.txt')):
            self.__saveCsv__(paramFileName)
        elif paramFileName.lower().endswith('.npy'):
            self.__saveNpy__(paramFileName)
        return()

    def load(self,paramFileName):
        if paramFileName.lower().endswith(('.csv', '.txt')):
            self.__loadCsv__(paramFileName)
        elif paramFileName.lower().endswith('.npy'):
            self.__loadNpy__(paramFileName)
        return()

    def compute_rbf(self):
        self.RBF=[]
        x = self.times
        if self.nSites == 1:
            for p in range(self.nParams):
                y = self.params[:,p]
                self.RBF.append(Rbf(x, y))
        else:
            for s in range(self.nSites):
                RBF_tmp = []
                for p in range(self.nParams):
                    y = self.params[:,p,s]
                    RBF_tmp.append(Rbf(x, y))
                self.RBF.append(RBF_tmp)


    def compute_param_over_time(self,Tf):
        self.compute_rbf()
        self.params_time = np.zeros((self.dataEnd+1,self.nParams,self.nSites)).squeeze()
        for p in range(self.nParams):
            if self.constant[p]:
                self.params_time[:,p] = self.params[0,p]
            elif self.nSites == 1:
                self.params_time[:,p] = self.RBF[p](range(self.dataEnd+1))
            else:
                for s in range(self.nSites):
                    if self.constantSites[s]:
                        self.params_time[:,p,s] = self.RBF[0][p](range(self.dataEnd+1))
                    else:
                        self.params_time[:,p,s] = self.RBF[s][p](range(self.dataEnd+1))

    def addPhase(self,ndays):
        self.nPhases += 1
        self.times        = np.append(self.times,ndays)
        self.params       = np.append(self.params,[self.params[-1]],axis=0)
        self.mask         = np.append(self.mask,[self.mask[-1]],axis=0)
        self.lower_bounds = np.append(self.lower_bounds,[self.lower_bounds[-1]],axis=0)
        self.upper_bounds = np.append(self.upper_bounds,[self.upper_bounds[-1]],axis=0)

            
    def getPhase(self,p,t):
        if self.constant[p]:
            phase = 0
        else:
            phase = self.nPhases-1
            for i, interval in enumerate(self.times[1:]):
                if ( t <= interval ):
                    phase = i
                    break
        return (phase)

    def atTime(self,t):
        params_time = np.zeros((self.nParams,self.nSites)).squeeze()
        if self.nSites==1:
            if self.dataEnd>0 and t>self.dataEnd:
                m = 1
                if len(self.scenario) > 0:
                    d,s = self.scenario.transpose()
                    i = np.searchsorted(d,t,side='right')-1
                    if i>=0:
                        if len(d)==1:
                            for q in range(self.nParams):
                                params_time[q] = np.maximum(self.scenario_extrapolator[q](t), 0)
                            return params_time
                        else:
                            t = d[0] - 1
                            m = s[i]
                params_time = np.array(self.params[-1])
                if type(self.degree)==int or self.degree == 'rbf':
                    for q in range(self.nParams):
                        if q==0:
                            params_time[q] = np.maximum(self.extrapolator[q](t) * m,0)
                        else:
                            params_time[q] = np.maximum(self.extrapolator[q](t),0)
                else:
                    params_time[0] = self.extrapolator(x=t) * m
            else:
                params_time = self.params_time[int(t)]

        else:
            if self.dataEnd>0 and t>self.dataEnd:
                for p in range(self.nSites):
                    m = 1
                    if len(self.scenario) > 0:
                        d,s = self.scenario[p].transpose()
                        i = np.searchsorted(d,t,side='right')-1
                        if i>=0:
                            if len(d) == 1:
                                for q in range(self.nParams):
                                    params_time[q,p] = np.maximum(self.scenario_extrapolator[p][q](t), 0)
                                return params_time
                            else:
                                t = d[0] - 1
                                m = s[i]
                    params_time[:,p] = np.array(self.params[-1,:,p])
                    if type(self.degree)==int or self.degree == 'rbf':
                        for q in range(self.nParams):
                            if q==0:
                                params_time[q,p] = np.maximum(self.extrapolator[p][q](t) * m,0)
                            else:
                                params_time[q,p] = np.maximum(self.extrapolator[p][q](t),0)
                    else:
                        params_time[0,p] = self.extrapolator[p](x=t) * m
            else:
                params_time = self.params_time[int(t)]

        return params_time

    def atPhase(self,i):
        return(self.params[ i , ...])

    def atSite(self,i): # works only if more than 1 site
        if self.nSites > 1:
            return(self.params[ ... , i ])
        return ()

    def forecast(self, DPC_time, Tf, deg, scenarios=None):
        if DPC_time>=Tf:
            return ()
        self.degree = deg
        self.dataEnd = int(DPC_time)
        tmp_times = np.concatenate(([0],self.times,[self.dataEnd]))
        if self.nSites == 1:
            if type(self.degree)==int:
                x = np.arange(self.dataEnd - deg, self.dataEnd+1,1)
                self.extrapolator = []
                for q in range(self.nParams):
                    try:
                        y = self.params_time[-(deg+1):,q]
                    except:
                        y = self.get()[-(deg+1):,q]
                    self.extrapolator.append(np.poly1d(np.polyfit(x,y,self.degree)))
            elif self.degree == 'exp':
                x = self.times[1:]
                y = self.get()[1:,0]
                EMG = EMGextrapol(x,y)
                self.extrapolator = functools.partial(EMG.eval,**EMG.best_values)
            elif self.degree == 'rbf':
                self.compute_rbf()
                self.extrapolator = self.RBF
        else:
            self.extrapolator = []
            if type(self.degree)==int:
                for p in range(self.nSites):
                    tmp_extrapolator = []
                    x = np.arange(self.dataEnd - deg, self.dataEnd+1,1)
                    for q in range(self.nParams):
                        try:
                            y = self.params_time[-(deg+1):,q,p]
                        except:
                            y = self.get()[-(deg+1):,q,p]
                        tmp_extrapolator.append(np.poly1d(np.polyfit(x,y,self.degree)))
                    self.extrapolator.append(tmp_extrapolator)
            elif self.degree == 'exp':
                x = self.times[1:]
                for p in range(self.nSites):
                    y = self.get()[1:,0,p]
                    EMG = EMGextrapol(x,y)
                    self.extrapolator.append(functools.partial(EMG.eval,**EMG.best_values))
            elif self.degree == 'rbf':
                self.compute_rbf()
                self.extrapolator = self.RBF

        if scenarios is not None:
            self.scenario = scenarios
            if self.nSites != 1:
                if len(scenarios.shape) == 2:
                    self.scenario = np.tile(self.scenario,(self.nSites,1,1))
        return ()

    def extrapolate_scenario(self):
        if self.nSites == 1:
            if self.scenario.shape[0] != 1:
                return()
            d,s = self.scenario.transpose()
            tmp_times = np.arange(d+1)
            if type(self.degree)==int:
                x = tmp_times[-(self.degree+1):]
                #x = tmp_times[-1:]
                self.scenario_extrapolator = []
                for q in range(self.nParams):
                    if q==0:
                        y = np.concatenate((self.get()[:,q],self.extrapolator[q](d-1)*s))
                    else:
                        y = np.concatenate((self.get()[:,q],self.extrapolator[q](d)))
                    self.scenario_extrapolator.append(np.poly1d(np.polyfit(x,y[-(self.degree+1):],self.degree)))
                    #self.scenario_extrapolator.append(np.poly1d(np.polyfit(x,y[-1:],0)))
        else:
            if self.scenario.shape[1] != 1:
                return()
            self.scenario_extrapolator = []
            for p in range(self.nSites):
                d,s = self.scenario[p].transpose()
                tmp_times = np.concatenate(([0],self.times,[self.dataEnd],d))
                if type(self.degree)==int:
                    x = tmp_times[-(self.degree+1):]
                    tmp_scenario_extrapolator = []
                    for q in range(self.nParams):
                        if q==0:
                            y = np.concatenate((self.get()[:,q,p],self.extrapolator[p][q](d-1)*s))
                        else:
                            y = np.concatenate((self.get()[:,q,p],self.extrapolator[p][q](d)))
                        tmp_scenario_extrapolator.append(np.poly1d(np.polyfit(x,y[-(self.degree+1):],self.degree)))
                    self.scenario_extrapolator.append(tmp_scenario_extrapolator)
        return ()

    def __saveCsv__(self,paramFileName):
        with open(paramFileName, "w") as f:
            print(f"[nParams]", file=f)
            print(self.nParams, file=f)
            print(f"[nSites]", file=f)
            print(self.nSites, file=f)
            print(f"[nPhases]", file=f)
            print(self.nPhases, file=f)
            print(f"[times]", file=f)
            if len(self.times) != 0:
                tmp = '\n'.join(map(lambda x: (self.dataStart + datetime.timedelta(days=int(x))).isoformat(), self.times))
                print(tmp, file=f)
            print(f"[constant]", file=f)
            if len(self.constant) != 0:
                tmp = ' '.join(('%f' % x).rstrip('0').rstrip('.') \
                         for x in self.constant)
                print(tmp, file=f)
            if len(self.constantSites) != 0:
                print(f"[constantSites]", file=f)
                tmp = ' '.join(('%f' % x).rstrip('0').rstrip('.') \
                         for x in self.constantSites)
                print(tmp, file=f)
            print(f"[Estimated]", file=f)
            print(int(self.estimated),file=f)
            print("", file=f)
            print(f"[params]", file=f)
            if len(self.params) != 0:
                if self.nSites==1:
                    tmp = '\n'.join(' '.join(np.format_float_positional(x,precision=8,pad_right=8).rstrip('0').rstrip('.') \
                        for x in y) for y in self.params)
                else:
                    tmp = '\n\n'.join('\n'.join(' '.join(np.format_float_positional(x,precision=8,pad_right=8).rstrip('0').rstrip('.') \
                        for x in y) for y in z) for z in np.moveaxis(self.params,-1,0))
                print(tmp, file=f)
            print("", file=f)
            print(f"[mask]", file=f)
            if len(self.mask) != 0:
                if self.nSites == 1:
                    tmp = '\n'.join(' '.join(('%f' % x).rstrip('0').rstrip('.') \
                         for x in y) for y in self.mask)
                else:
                    tmp = '\n\n'.join('\n'.join(' '.join(('%f' % x).rstrip('0').rstrip('.') \
                         for x in y) for y in z) for z in np.moveaxis(self.mask,-1,0))
                print(tmp, file=f)
            print("", file=f)
            print(f"[l_bounds]", file=f)
            if len(self.lower_bounds) != 0:
                if self.nSites == 1:
                    tmp = '\n'.join(' '.join(('%f' % x).rstrip('0').rstrip('.') \
                         for x in y) for y in self.lower_bounds)
                else:
                    tmp = '\n\n'.join('\n'.join(' '.join(('%f' % x).rstrip('0').rstrip('.') \
                         for x in y) for y in z) for z in np.moveaxis(self.lower_bounds,-1,0))
                print(tmp, file=f)
            print("", file=f)
            print(f"[u_bounds]", file=f)
            if len(self.upper_bounds) != 0:
                if self.nSites == 1:
                    tmp = '\n'.join(' '.join(('%f' % x).rstrip('0').rstrip('.') \
                         for x in y) for y in self.upper_bounds)
                else:
                    tmp = '\n\n'.join('\n'.join(' '.join(('%f' % x).rstrip('0').rstrip('.') \
                         for x in y) for y in z) for z in np.moveaxis(self.upper_bounds,-1,0))
                print(tmp, file=f)

    def __saveNpy__(self,paramFileName):
        with open(paramFileName, 'wb') as f:
            np.savez(f, nParams = self.nParams, \
                        nSites = self.nSites, \
                        nPhases = self.nPhases, \
                        estimated = self.estimated,\
                        times = self.times, \
                        constant = self.constant, \
                        params = self.params, \
                        mask = self.mask, \
                        lower_bounds = self.lower_bounds, \
                        upper_bounds = self.upper_bounds )

    def __loadCsv__(self,paramFileName):
        with open(paramFileName, 'rb') as f:
            content = f.readlines()

        self.nParams = int(readSection(content,b'[nParams]',1))
        try:
            self.nSites = int(readSection(content,b'[nSites]',1))
        except:
            self.nSites = 1
        self.nPhases = int(readSection(content,b'[nPhases]',1))

        tmp = readTimes(content, b'[times]', self.nPhases)
        self.times = np.reshape([int((x-self.dataStart).days) for x in tmp],self.nPhases)
        try:
            self.constant = np.reshape( \
            readSection(content, b'[constant]', self.nParams), \
            self.nParams)
        except:
            self.constant = np.zeros(self.nParams)
        if self.nSites > 1:
            try:
                self.constantSites = np.reshape( \
                readSection(content, b'[constantSites]', self.nParams), \
                self.nParams)
            except:
                self.constantSites = np.zeros(self.nParams)
        try:
            self.estimated = bool(readSection(content,b'[Estimated]',1))
        except:
            self.estimated = False
        nParams = self.nParams * self.nPhases if not self.estimated else self.nParams * self.nPhases * self.nSites
        self.params = readSection(content, b'[params]', nParams)
        if not self.estimated:
            self.params = np.tile(self.params, (self.nSites,1))
        self.params = np.reshape(self.params, (self.nSites, self.nPhases, self.nParams))
        self.params=np.moveaxis(self.params,0,-1).squeeze()
        self.mask = readSection(content, b'[mask]', nParams)
        if not self.estimated:
            self.mask = np.tile(self.mask, (self.nSites,1))
        self.mask = np.reshape(self.mask, (self.nSites, self.nPhases, self.nParams))
        self.mask=np.moveaxis(self.mask,0,-1).squeeze()
        self.lower_bounds = readSection(content, b'[l_bounds]', nParams)
        if not self.estimated:
            self.lower_bounds = np.tile(self.lower_bounds, (self.nSites,1))
        self.lower_bounds = np.reshape(self.lower_bounds, (self.nSites,self.nPhases, self.nParams))
        self.lower_bounds = np.moveaxis(self.lower_bounds,0,-1).squeeze()
        self.upper_bounds = readSection(content, b'[u_bounds]', nParams)
        if not self.estimated:
            self.upper_bounds = np.tile(self.upper_bounds, (self.nSites,1))
        self.upper_bounds = np.reshape(self.upper_bounds, (self.nSites, self.nPhases, self.nParams))
        self.upper_bounds = np.moveaxis(self.upper_bounds,0,-1).squeeze()

    def __loadNpy__(self,paramFileName):
        with open(paramFileName, 'rb') as f:
            data = np.load(f)
            self.nParams = data['nParams']
            try:
                self.nSites = data['nSites']
            except:
                self.nSites = 1
            self.nPhases = data['nPhases']
            self.times = data['times']
            try:
                self.constant = data['constant']
            except:
                self.constant= np.zeros(self.nPhases)
            try:
                self.estimated = data['estimated']
            except:
                self.estimated = False
            self.params = data['params']
            self.mask = data['mask']
            self.lower_bounds = data['lower_bounds']
            self.upper_bounds = data['upper_bounds']
