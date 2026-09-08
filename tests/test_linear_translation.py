"""Physical-coordinate contracts for normalized rigid/affine translations."""
import unittest

import numpy as np
import SimpleITK as sitk
import torch

from fireants.io import Image, BatchedImages
from fireants.registration.affine import AffineRegistration
from fireants.registration.rigid import RigidRegistration
from fireants.registration.translation import translation_parameters


def images(dims=3, unit=1., size=36, shift=False, origin=0., angle=0.):
    shape=(size,)*dims
    coords=np.indices(shape)
    data=np.exp(-sum((coords[i]-(size*.4+i))**2 for i in range(dims))/45).astype('float32')
    if shift: data=np.roll(data,2,axis=-1)
    itk=sitk.GetImageFromArray(data)
    itk.SetSpacing([unit*35/(size-1)]*dims)
    itk.SetOrigin([origin*unit]*dims)
    direction=np.eye(dims)
    direction[:2,:2]=[[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]
    itk.SetDirection(direction.ravel())
    return BatchedImages([Image(itk,device='cpu')])


class NormalizedTranslationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_radius_independent_of_origin_orientation_and_resolution(self):
        for dims in [2,3]:
            for size,origin,angle in [(36,0,0),(71,100,.3)]:
                f=images(dims,size=size,origin=origin,angle=angle)
                radius,rate=translation_parameters(f,True,None,.01)
                torch.testing.assert_close(radius,torch.tensor([[17.5]]),atol=2e-5,rtol=1e-5)
                self.assertEqual(rate,.01)

    def test_initial_physical_matrix_is_preserved(self):
        for cls in [RigidRegistration,AffineRegistration]:
            for dims in [2,3]:
                for around_center in [False,True]:
                    with self.subTest(cls=cls.__name__,dims=dims,center=around_center):
                        f=images(dims,origin=80,angle=.2)
                        matrix=torch.eye(dims+1)[None]
                        angle=.15
                        matrix[0,:2,:2]=torch.tensor([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
                        matrix[0,:dims,-1]=torch.arange(dims)+2.
                        init=({'init_translation':matrix[:,:dims,-1], 'init_moment':matrix[:,:dims,:dims]} if cls is RigidRegistration else {'init_rigid':matrix})
                        for normalized in [False,True]:
                            reg=cls(scales=[1],iterations=[0],fixed_images=f,moving_images=f,loss_type='mse',
                                    normalize_translation=normalized,around_center=around_center,**init)
                            actual=reg.get_rigid_matrix() if cls is RigidRegistration else reg.get_affine_matrix()
                            torch.testing.assert_close(actual,matrix,atol=2e-5,rtol=1e-5)

    def test_optimization_independent_of_physical_units(self):
        for dims in [2,3]:
            for cls in [RigidRegistration,AffineRegistration]:
                for optimizer in ['Adam','SGD']:
                    with self.subTest(dims=dims,cls=cls.__name__,optimizer=optimizer):
                        grids=[]
                        for unit in [1.,1000.]:
                            fixed,moving=images(dims,unit=unit),images(dims,unit=unit,shift=True)
                            reg=cls(scales=[1],iterations=[8],fixed_images=fixed,moving_images=moving,
                                    loss_type='mse',optimizer=optimizer,optimizer_lr=.01,
                                    normalize_translation=True,translation_lr=.007,progress_bar=False)
                            reg.optimize();grids.append(reg.get_warped_coordinates(fixed,moving).detach())
                        torch.testing.assert_close(*grids,atol=2e-5,rtol=2e-5)

    def test_invalid_rates_rejected(self):
        f=images()
        for cls in [RigidRegistration,AffineRegistration]:
            for rate in [-1,0,float('nan'),float('inf')]:
                with self.assertRaisesRegex(ValueError,'finite and positive'):
                    cls(scales=[1],iterations=[0],fixed_images=f,moving_images=f,normalize_translation=True,translation_lr=rate)
            with self.assertRaisesRegex(ValueError,'normalize_translation'):
                cls(scales=[1],iterations=[0],fixed_images=f,moving_images=f,translation_lr=.01)


if __name__=='__main__': unittest.main()
