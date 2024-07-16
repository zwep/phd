
import sys
import os
import re
import numpy as np


def filename_extcase(fn):
    pn, ext = os.path.splitext(fn)
    bn = os.path.basename(pn)
    if bn + ext.lower() in os.listdir(os.path.dirname(fn)):
        return pn + ext.lower()
    if bn + ext.upper() in os.listdir(os.path.dirname(fn)):
        return pn + ext.upper()
    return ''

def oset(seq, idfun = None):
    if idfun is None:

        def idfun(x):
            return x

    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        if marker in seen:
            continue
        seen[marker] = 1
        result.append(item)

    return result


def oset_sorted(seq, idfun = None):
    if idfun is None:

        def idfun(x):
            return x

    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        if marker in seen:
            continue
        seen[marker] = 1
        result.append(item)

    result = sorted(result, key=lambda x: float(x))
    return result


def readRec(filename, cur_loc, rescale_type):
    """
        read from the REC file using header data from the PAR or XML file
    """
    filename, _ = os.path.splitext(filename)
    filename_par = filename_extcase(filename + '.PAR')
    filename_xml = filename_extcase(filename + '.XML')
    filename_rec = filename_extcase(filename + '.REC')
    headerType = 0
    if os.path.exists(filename_rec) and os.path.exists(filename_xml):
        hdr = readXML(filename_xml)
        headerType = 0
    else:
        if os.path.exists(filename_rec) and os.path.exists(filename_par):
            hdr = readPar(filename_par)
            headerType = 1
        else:
            print("ReadPhilips():readRec():ERROR: PAR, XML, and/or REC path doesn't exist.")
            return 1
        if type(filename_rec) is not str:
            print('Input filename is not a string.')
            sys.exit(1)
        if os.path.splitext(filename_rec)[1] not in ('.REC', '.rec'):
            print('Input filename is not a .REC file')
            sys.exit(1)
        try:
            fil = open(filename_rec, 'rb')
        except IOError:
            print('cannot open .rec file ', filename_rec)
            sys.exit(1)

        dynamics = oset_sorted(hdr['Dynamic'])
        phases = oset_sorted(hdr['Phase'])
        echoes = oset_sorted(hdr['Echo'])
        loc_case = np.ones(len(hdr['Slice'])) > 0
        if cur_loc != -1:
            loc_case = hdr['Slice'] == np.str(cur_loc)
        read_ind = loc_case.nonzero()[0]
        locations = oset_sorted(hdr['Slice'][read_ind])
        imtypes = oset(hdr['Type'])
        seqtypes = oset(hdr['Sequence'])
        ndyn = len(dynamics)
        ncard = len(phases)
        necho = len(echoes)
        nlocs = len(locations)
        nimtype = len(oset(imtypes))
        nsqtype = len(oset(seqtypes))
        version = '4'
        try:
            diffvalues = oset_sorted(hdr['BValue'])
            gradorients = oset_sorted(hdr['Grad Orient'])
            ndiff = len(diffvalues)
            ngrad = len(gradorients)
            version = '4.1'
        except:
            pass

        try:
            labtypes = oset(hdr['Label Type'])
            nlabel = len(labtypes)
            version = '4.2'
        except:
            pass

        print('Debug: ndyn/ncard/nloc', ndyn, ncard, nlocs)
        contrasts = int(np.ceil(len(hdr['Index']) / float(ndyn * ncard * nlocs)))
        expected_contrasts = necho * nimtype * nsqtype
        if headerType == 0 or version in ('4.1', '4.2'):
            expected_contrasts = expected_contrasts * ndiff * ngrad
        if headerType == 0 or version == '4.2':
            expected_contrasts = expected_contrasts * nlabel
        flatten = 'False'
        if expected_contrasts != contrasts or contrasts % expected_contrasts != 0:
            flatten = True
        recx = int(hdr['Resolution X'][0])
        recy = int(hdr['Resolution Y'][0])
        if flatten is True:
            outshape = [
             contrasts, ndyn, ncard, nlocs, recy, recx]
            data_string = np.array(['cntrst', 'dyn', 'card', 'loc', 'x', 'y'])
        else:
            if headerType == 0 or version == '4.2':
                outshape = [
                 nimtype, nsqtype, nlabel, ngrad, ndiff, ndyn, ncard,
                 necho, nlocs, recy, recx]
                data_string = np.array(['imtype', 'seq', 'lab', 'grad', 'bval', 'dyn',
                 'card', 'echo', 'loc', 'x', 'y'])
            else:
                if version == '4.1':
                    outshape = [
                     nimtype, nsqtype, ngrad, ndiff, ndyn, ncard, necho, nlocs,
                     recy, recx]
                    data_string = np.array(['imtype', 'seq', 'grad', 'bval', 'dyn', 'card',
                     'echo', 'loc', 'x', 'y'])
                else:
                    outshape = [
                     nimtype, nsqtype, ndyn, ncard, necho, nlocs, recy, recx]
                    data_string = np.array(['imtype', 'seq', 'dyn', 'card', 'echo', 'loc',
                     'x', 'y'])
    data_concat = np.zeros(outshape, dtype=np.float32)
    for idx in hdr['Index'][read_ind]:
        idx = int(idx)
        dyn = dynamics.index(hdr['Dynamic'][idx])
        card = phases.index(hdr['Phase'][idx])
        echo = echoes.index(hdr['Echo'][idx])
        loc = locations.index(hdr['Slice'][idx])
        if headerType == 0 or version in ('4.1', '4.2'):
            bval = diffvalues.index(hdr['BValue'][idx])
            grad = gradorients.index(hdr['Grad Orient'][idx])
        if headerType == 0 or version == '4.2':
            lab = labtypes.index(hdr['Label Type'][idx])
        typ = imtypes.index(hdr['Type'][idx])
        seq = seqtypes.index(hdr['Sequence'][idx])
        pixel_bytes = int(hdr['Pixel Size'][idx]) // 8
        data_type = np.uint16
        if pixel_bytes == 1:
            data_type = np.uint8
        elif pixel_bytes == 2:
            data_type = np.uint16
        size_bytes = recx * recy * pixel_bytes
        ri = float(hdr['Rescale Intercept'][idx])
        rs = float(hdr['Rescale Slope'][idx])
        ss = float(hdr['Scale Slope'][idx])
        cont = int(idx // (ndyn * ncard)) % contrasts
        offset = idx * size_bytes
        fil.seek(offset)
        data = fil.read(size_bytes)
        data = np.fromstring(data, dtype=data_type)
        if rescale_type == 0:
            data = (data * rs + ri) / (rs * ss)
        elif rescale_type == 1:
            data = data * rs + ri
        try:
            data.shape = [
             recx, recy]
            if flatten is True:
                data_concat[cont, dyn, card, loc, 0:recx, 0:recy] = data
            else:
                if headerType == 0 or version == '4.2':
                    data_concat[typ, seq, lab, grad, bval, dyn, card, echo, loc, 0:recx, 0:recy] = data
                else:
                    if version == '4.1':
                        data_concat[typ, seq, grad, bval, dyn, card, echo, loc, 0:recx, 0:recy] = data
                    else:
                        data_concat[typ, seq, dyn, card, echo, loc, 0:recx, 0:recy] = data
        except:
            print('\n\tERROR: index out of range:')
            print('\tdata->dimensions:' + str(data_concat.shape))
            if flatten is True:
                print('\tcurrent index   :' + str([cont, dyn, card, loc]))
            else:
                if headerType == 0 or headerType == '4.2':
                    print('\tcurrent index   :' + str([typ, seq, lab, grad, bval,
                     dyn, card, echo, loc]))
                else:
                    if headerType == '4.1':
                        print('\tcurrent index   :' + str([typ, seq, grad, bval, dyn,
                         card, echo, loc]))
                    else:
                        print('\tcurrent index   :' + str([typ, seq, dyn, card, echo,
                         loc]))
            print(' ')
            raise

    fil.close()
    data_labels = data_string[(np.array(data_concat.shape[0:len(data_string)]) > 1).nonzero()[0]]
    return (
     data_concat, hdr, data_labels)


def readXML(filename):
    """
        Parse a *.xml header accompanying the *.rec image file.
    """
    import xml.etree.ElementTree as et
    filename = filename_extcase(filename)
    if type(filename) is not str:
        print('Input filename is not a string.')
        sys.exit(1)
    if os.path.splitext(filename)[1] not in ('.XML', '.xml'):
        print('Input filename is not a .XML file')
        sys.exit(1)
    try:
        tree = et.parse(filename)
        root = tree.getroot()
    except IOError:
        print('cannot open', filename)

    tab = [line.attrib['Name'] for line in root[1][0].iter('Attribute')]
    text = str([line.text for line in root[1].iter('Attribute')]).strip('[]')
    text = re.split(', +', text)
    text = [line.strip("'") for line in text]
    try:
        loc = list(zip(*[iter(text)] * len(tab)))
    except:
        print('coding error: defined table length not equal to number of image attributes')

    info = dict()
    info['headerType'] = '.xml'
    for label_idx in range(len(tab)):
        label = tab[label_idx]
        if label_idx < len(loc[0]):
            vals = [line[label_idx] for line in loc]
            info[label] = np.array(vals)

    return info


def readPar(filename):
    """
        Parse a *.PAR header accompanying the *.REC image file.
    """
    filename = filename_extcase(filename)
    if type(filename) is not str:
        print('Input filename is not a string.')
        sys.exit(1)
    if os.path.splitext(filename)[1] not in ('.PAR', '.par'):
        print('Input filename is not a .PAR file')
        sys.exit(1)
    try:
        fil = open(filename, 'r')
    except IOError:
        print('cannot open .par file ', filename)
        sys.exit(1)

    lines = fil.readlines()
    fil.close()
    loc = [line for line in lines if line[0] not in ('#', '.') and len(line) > 2]
    loc = [line.strip() for line in loc]
    loc = [re.split(' +', line) for line in loc]
    tab = 'Slice,Echo,Dynamic,Phase,Type,Sequence,Index,Pixel Size,' + 'Scan Percentage,Resolution X,Resolution Y,Rescale Intercept,' + 'Rescale Slope,Scale Slope,Window Center,Window Width,' + 'Angulation AP,Angulation FH,Angulation RL,Offcenter AP,' + 'Offcenter FH,Offcenter RL,Slice Thickness,Slice Gap,' + 'Display Orientation,Slice Orientation,fMRI Status Indication,' + 'Image Type Ed Es,Pixel Spacing_0,Pixel Spacing_1,Echo Time,' + 'Dyn Scan Begin Time,Trigger Time,Diffusion B Factor,No Averages,' + 'Image Flip Angle,Cardiac Frequency,Min RR Interval,' + 'Max RR Interval,TURBO Factor,Inversion Delay,BValue,Grad Orient,' + 'Contrast Type,Diffusion Anisotropy Type,Diffusion AP,Diffusion FH,' + 'Diffusion RL,Label Type'
    tab = re.split(',', tab)
    info = dict()
    info['headerType'] = '.par'
    for label_idx in range(len(tab)):
        label = tab[label_idx]

        if label_idx < len(loc[0]):
            vals = [line[label_idx] for line in loc]
            info[label] = np.array(vals)

    return info


