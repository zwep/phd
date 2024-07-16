
"""
Overload the funciton with a new method so that we DONT get the relative phases...

Could add it as an option of course.. but nah.
"""

import sigpy.mri


class EspiritCalib(sigpy.mri.app.EspiritCalib):
    def _output(self):
        xp = self.device.xp
        with self.device:
            # Dont normalize phase with respect to first channel
            mps = self.mps.T[0]
            # mps *= xp.conj(mps[0] / xp.abs(mps[0]))

            # Crop maps by thresholding eigenvalue
            max_eig = self.alg.max_eig.T[0]
            mps *= max_eig > self.crop

        if self.output_eigenvalue:
            return mps, max_eig
        else:
            return mps
