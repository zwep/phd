"""
In the PPE we see this piece of code

for (angle_ix = 0; angle_ix < MG_GC_UACQrcd_par( 15 )->value_ptr->to_int; )
{
    float	angle;
    angle  = angle_ix * angle_factor + angle_offset;
    sin_table_aptr[ angle_ix ] = (short) (sin( angle ) * MMIFFE_SIN_SCALE);
    cos_table_aptr[ angle_ix ] = (short) (cos( angle ) * MMIFFE_SIN_SCALE);
    angle_ix++;
    if ( alternate )
    {
        angle  = angle_ix * angle_factor + angle_offset;
        sin_table_aptr[ angle_ix ] = -(short) (sin( angle ) * MMIFFE_SIN_SCALE);
        cos_table_aptr[ angle_ix ] = -(short) (cos( angle ) * MMIFFE_SIN_SCALE);
        angle_ix++;
    }
}

"""

import numpy as np
import helper.plot_class as hplotc
import matplotlib.pyplot as plt
import helper.reconstruction as hrecon

n_spokes = 50

angle_from_recon = hrecon.get_angle_spokes(n_spokes)
plt.plot(angle_from_recon)

# I dont know if this Angle Factor is the one we need
angle_factor = 2 * np.pi / n_spokes
angle_offset = 0
sin_table_aptr = []
cos_table_aptr = []
alternate = False
for angle_ix in np.arange(n_spokes):
    angle = angle_ix * angle_factor + angle_offset
    sin_table_aptr.append(np.sin(angle))
    cos_table_aptr.append(np.cos(angle))
    if alternate:
        angle = angle_ix * angle_factor+angle_offset
        sin_table_aptr.append(-np.sin(angle))
        cos_table_aptr.append(-np.cos(angle))


fig, ax = plt.subplots()
ax.scatter(sin_table_aptr, cos_table_aptr)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
