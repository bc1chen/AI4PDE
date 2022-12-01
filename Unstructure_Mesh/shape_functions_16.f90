
        program main ! ann filter and simple code
        implicit none
! integers representing the length of arrays...
! totele=no of elements,nloc=no of nodes per element, totele_nloc=totele*nloc
! sngi=no of surface quadrature points of the faces - this is set to the max no of all faces.
! ngi=no of surface quadrature points of the faces.
! ndim=no of dimensions - including time possibly, nface=no of faces of each elemenet, nc no of fields to solve for.
        
! local variables...
! 2d:
! nx=no of nodes across, ny=no of nodes up. 
        integer, parameter :: nx=11,ny=11, nonods=nx*ny, totele=(nx-1)*(ny-1)
        integer, parameter :: nloc=4,sngi=3, ngi=9, ndim=2, &
                              nface=4, max_face_list_no=2
        integer, parameter :: ntime = 10
        real, parameter :: dx=1.0, dy=1.0, dt=0.1 
! example of node ordering...
!      7-----8-----9
!      !     !     !
!      !     !     !        
!      4-----5-----6
!      !     !     !
!      !     !     !        
!      1-----2-----3
        real b(nonods), ml(nonods)
        real k(nonods), sig(nonods), s(nonods), t_new(nonods), t_old(nonods)
        real u(ndim,nonods)
        real x_all(ndim,nloc*totele)
        integer fina(nonods+1) 
        integer ndglno(nloc*totele)
! filters for ann:
        integer, allocatable :: cola(:) 
        real, allocatable :: a(:),a_filter(:) 
        integer npoly,ele_type, iloc,count, ncola
        integer enx_i, eny_j, ele, i,j, ii,jj, iii,jjj, inod, jnod, itime
! 
        ncola=9*nonods ! this is the maximum size of ncola it will be smaller. 
        allocate(a(ncola),a_filter(ncola))
        allocate(cola(ncola))
! 
        print *,'here in main'
!        stop 23
! 
        call set_up_grid_problem(t_old, t_new, x_all, ndglno, fina, cola, &
                                 dx, dy, ncola, nx, ny, nloc, ndim)

        npoly=1
        ele_type=1
        k=0.0; sig=0.0; s=0.0; u(1,:)=1.0; u(2,:)=1.0
!        do inod=1,nonods
!           if(
!        end do
! in python:
! a, b = u2r.get_fe_matrix_eqn(x_all, u, k, sig, s, fina,cola, ncola, ndglno, nonods,totele,nloc, ndim, ele_type)
! ml = the lumped mass 
        call get_fe_matrix_eqn(a,b, ml, k, sig, s, u, x_all, &
                               fina,cola, ndglno,  &
                               ele_type, ndim, totele,nloc, ncola,nonods) 
! this subroutine finds the matrix eqns A T=b - that is it forms matrix a and vector b and the soln vector is T 
! in python:
! a_filter = get_filter_matrix( a, ml, dt, fina,cola, ncola,nonods)
! This subroutine finds the matrix eqns a_filter = -M_L^{-1} ( -M_L + dt* A). 
! Time stepping can be realised with T^{n+1} = a_filter * T^n 
        call get_filter_matrix(a_filter, a, ml, dt,  &
                               fina,cola, ncola,nonods)

! in python:
! t_new = time_step_filter_matrix( t_old, a_filter, fina,cola, ncola,nonods)
        do itime=1,ntime
! set bcs...
           do j=1,ny
           do i=1,nx
              inod=(j-1)*nx+i
              if((i==1).or.(i==nx)) t_old(inod)=0.0
              if((j==1).or.(j==ny)) t_old(inod)=0.0
           end do
           end do

           call time_step_filter_matrix(t_new, t_old, a_filter,   &
                                        fina,cola, ncola, nonods)
           t_old=t_new
        end do
        do j=ny,1,-1
           print *,'j,t_new( (j-1)*nx+1: (j-1)*nx +nx):',j
           print *,t_new( (j-1)*nx+1: (j-1)*nx +nx)
        end do
        stop 2922
        end program main
! 
! 
! 
! 
        subroutine another_main2 ! ann filter and simple code
        implicit none
! integers representing the length of arrays...
! totele=no of elements,nloc=no of nodes per element, totele_nloc=totele*nloc
! sngi=no of surface quadrature points of the faces - this is set to the max no of all faces.
! ngi=no of surface quadrature points of the faces.
! ndim=no of dimensions - including time possibly, nface=no of faces of each elemenet, nc no of fields to solve for.
        
! local variables...
! 2d:
        integer, parameter :: totele=1,nloc=4,sngi=3, ngi=9, ndim=2, &
                              nface=4, max_face_list_no=2
! 3d:
!        integer, parameter :: totele=1,nloc=8,sngi=9, ngi=27, ndim=3, &
!                              nface=6, max_face_list_no=4
        real :: n(ngi,nloc), nlx(ngi,ndim,nloc), weight(ngi)
        real :: face_sn(sngi,nloc,nface), face_sn2(sngi,nloc,max_face_list_no)
        real :: face_snlx(sngi,ndim,nloc,nface), face_sweigh(sngi,nface) 
! local variables...
        real, allocatable :: nx(:,:,:),detwei(:),inv_jac(:,:,:)
        real, allocatable :: nxnx(:,:), nnx(:,:,:)
! filters for ann:
        real, allocatable :: afilt_nxnx(:,:,:),  afilt_nnx(:,:,:,:)
        real, allocatable :: x_loc(:,:), x_all(:,:)
        integer npoly,ele_type,ele, iloc,jloc,idim
        integer ifil,jfil,kfil, iiblock, jjblock, kkblock, i,j,k, gi
        real rnorm

        allocate(nx(ngi,ndim,nloc),detwei(ngi),inv_jac(ngi,ndim,ndim))
        allocate(nxnx(nloc,nloc),nnx(ndim,nloc,nloc))
        allocate(x_loc(ndim,nloc)) 

!         stop 3382
        npoly=1
        ele_type=1

! form the shape functions...
        print *,'going into get_shape_funs_with_faces'
        call get_shape_funs_with_faces(n, nlx, weight,  &
               nloc, sngi, ngi, ndim, nface,max_face_list_no, &
               face_sn, face_sn2, face_snlx, face_sweigh, &
               npoly,ele_type) 

        allocate(x_all(ndim,totele*nloc)) 
! 
        if(ndim==2) then
          x_all(:,1)=(/-1.0,-1.0/)
          x_all(:,2)=(/+1.0,-1.0/)
          x_all(:,3)=(/-1.0,+1.0/)
          x_all(:,4)=(/+1.0,+1.0/)
        endif
        if(ndim==3) then
          x_all(:,1)=(/-1.0,-1.0,-1.0/)
          x_all(:,2)=(/+1.0,-1.0,-1.0/)
          x_all(:,3)=(/-1.0,+1.0,-1.0/)
          x_all(:,4)=(/+1.0,+1.0,-1.0/)
! 2nd layer...
          x_all(:,5)=(/-1.0,-1.0,+1.0/)
          x_all(:,6)=(/+1.0,-1.0,+1.0/)
          x_all(:,7)=(/-1.0,+1.0,+1.0/)
          x_all(:,8)=(/+1.0,+1.0,+1.0/)
        endif

! obtain 
! for filters...
        do ele = 1, totele ! VOLUME integral
             x_loc(:,:) = x_all(:,(ele-1)*nloc+1:ele*nloc)
             call det_nlx( x_loc, n, nlx, nx, detwei, weight, ndim, nloc, ngi, INV_JAC )

             nxnx=0.0
             nnx=0.0
             do iloc=1,nloc
             do jloc=1,nloc
                   do idim=1,ndim
          nxnx(iloc,jloc) = nxnx(iloc,jloc) + sum( nx(:,idim,iloc)*nx(:,idim,jloc)*detwei(:) )
          nnx(idim,iloc,jloc) = sum( n(:,iloc)*nx(:,idim,jloc)*detwei(:) )
                   end do
!                   do idim=1,1
!                      nxnx(iloc,jloc) = nxnx(iloc,jloc) + sum( nx(:,idim,iloc)*nx(:,idim,jloc)*detwei(:) )
!                   end do
             end do
             end do
        end do
! 
! now calculate filters:
        print *,'detwei:',detwei
        print *,'weight:',weight
        print *,' '
        do gi=1,ngi
          print *,'gi,idim,n(gi,:):',gi,n(gi,:)
        end do
        print *,' '
        do idim=1,1!ndim
        do gi=1,1!ngi
!          if(idim==1) print *,'gi,idim,n(gi,:):',gi,idim,n(gi,:)
          print *,'gi,idim,nlx(gi,idim,:):',gi,idim,nlx(gi,idim,:)
          iloc=1
!          print *,'iloc,nnx(idim,iloc,:):',iloc,nnx(idim,iloc,:)
        end do
        end do
        print *,' '
        do idim=1,1!ndim
        do gi=1,1!ngi
!          if(idim==1) print *,'gi,idim,n(gi,:):',gi,idim,n(gi,:)
!          print *,'gi,idim,nlx(gi,idim,:):',gi,idim,nlx(gi,idim,:)
          iloc=1
          print *,'iloc,nnx(idim,iloc,:):',iloc,nnx(idim,iloc,:)
        end do
        print *,' '
        end do
!        stop 211
!        do iloc=1,nloc
!          print *,'iloc,nnx(1,iloc,:):',iloc,nnx(1,iloc,:)
!        end do
        allocate(afilt_nxnx(3,3,1+2*(ndim-2)), afilt_nnx(ndim,3,3,1+2*(ndim-2)) )
        afilt_nxnx=0.0
        afilt_nnx =0.0
        do iiblock=1,2
        do jjblock=1,2
        do kkblock=1,ndim-1
           do ifil=1,2
           do jfil=1,2
           do kfil=1,ndim-1
!              do iidim2=1,2
!              do jjdim2=1,2
!              do kkdim2=1,ndim-1
                 iloc=1 + (2-iiblock) +(2-jjblock)*2   + (ndim-2)*(2-kkblock)*4
!                 jloc=iiblock +(jjblock-1)*2   + (kkblock-1)*4
                 jloc=ifil +(jfil-1)*2   + (kfil-1)*4
!                 jloc=iidim2 +(jjdim2-1)*2 + (kkdim2-1)*4
                 i=ifil + (iiblock-1)*1
                 j=jfil + (jjblock-1)*1
                 k=kfil + (kkblock-1)*1
                 afilt_nxnx(i,j,k)  = afilt_nxnx(i,j,k)  + nxnx(iloc,jloc)
                 afilt_nnx(:,i,j,k) = afilt_nnx(:,i,j,k) + nnx(:,iloc,jloc)
!              end do
!              end do
!              end do
           end do
           end do
           end do
        end do
        end do
        end do
! 
        do jloc=1,nloc
           print *,'jloc,nxnx(:,jloc):',jloc,nxnx(:,jloc)
        end do
! 
        do k=1,1+2*(ndim-2)
        do j=1,3
           print *,'k,j,afilt_nxnx(:,j,k):',k,j,afilt_nxnx(:,j,k)
        end do
        end do
!
! get rid of small values - round off error is an issue here. 
        do idim=1,ndim
           do k=1,1+2*(ndim-2)
           do j=1,3
           do i=1,3
             if( abs( afilt_nnx(idim,i,j,k))<7.e-5) afilt_nnx(idim,i,j,k)=0.0
           end do
           end do
           end do
        end do
! 
        do idim=1,ndim
           print *,' '
           do k=1,1+2*(ndim-2)
           do j=1,3
       print *,'idim,k,j,afilt_nnx(idim,:,j,k):',idim,k,j,afilt_nnx(idim,:,j,k)
           end do
           end do
        end do
! 
! normalise
        print *,' '
        print *,'normalised filter for diffusion (pre-multiplied by 1/dx^2):'
        if(ndim==2) rnorm=4.0/afilt_nxnx(2,2,1)
        if(ndim==3) rnorm=6.0/afilt_nxnx(2,2,2)
        do k=1,1+2*(ndim-2)
        do j=1,3
           print *,'k,j,afilt_nxnx(:,j,k):',k,j,afilt_nxnx(:,j,k)*rnorm
        end do
        end do
        print *,'sum( abs(afilt_nxnx(:,:,1:1+(ndim-2)*2))*rnorm ):',  &
                 sum( abs(afilt_nxnx(:,:,1:1+(ndim-2)*2))*rnorm )

        print *,' '
        print *,'normalised filter for advection (pre-multiplied by 1/dx):'
        do idim=1,ndim
           if(ndim==2) rnorm=1.0/sum(abs(afilt_nnx(idim,:,:,1)))
           if(ndim==3) rnorm=1.0/sum(abs(afilt_nnx(idim,:,:,:)))
           do k=1,1+2*(ndim-2)
           do j=1,3
              print *,'idim,k,j,afilt_nnx(idim,:,j,k):', &
                       idim,k,j,afilt_nnx(idim,:,j,k)*rnorm
           end do
           end do
           print *,' '
        end do
        stop 382
        end subroutine another_main2 
! 
! 
! 
! 
! in python: 
! t_old, t_new, x_all, ndglno, fina, cola  = set_up_grid_problem(dx, dy, ncola, nx, ny, nloc, ndim)
        subroutine set_up_grid_problem(t_old, t_new, x_all, ndglno, fina, cola, &
                                       dx, dy, ncola, nx, ny, nloc, ndim) 
        implicit none
! set up grid
! integers representing the length of arrays...
! totele=no of elements,nloc=no of nodes per element, totele_nloc=totele*nloc
! sngi=no of surface quadrature points of the faces - this is set to the max no of all faces.
! ngi=no of surface quadrature points of the faces.
! ndim=no of dimensions - including time possibly, nface=no of faces of each elemenet, nc no of fields to solve for.
        
! local variables...
! 2d:
! nx=no of nodes across, ny=no of nodes up. 
        integer, intent( in ) :: ncola, nx, ny, nloc, ndim
!        integer, parameter :: nonods=nx*ny, totele=(nx-1)*(ny-1)
        real, intent( out ) :: t_old(nx*ny), t_new(nx*ny), x_all(ndim,(nx+1)*(ny+1)*nloc) 
        integer, intent( out ) ::  ndglno(nloc*(nx+1)*(ny+1)), fina(nx*ny+1), cola(ncola) 
        real, intent( in ) :: dx, dy
! example of node ordering...
!      7-----8-----9
!      !     !     !
!      !     !     !        
!      4-----5-----6
!      !     !     !
!      !     !     !        
!      1-----2-----3
! filters for ann:
! local variables...
        integer npoly,ele_type, iloc,count
        integer enx_i, eny_j, ele, i,j, ii,jj, iii,jjj, inod, jnod, itime
! ! 
        print *,'here in setup grid'
!        stop 23
! 

        t_old=0.0
        do enx_i=1,nx-1
        do eny_j=1,ny-1
           ele=(eny_j-1)*(nx-1)+enx_i
!
           do ii=1,2
           do jj=1,2

              iloc=(jj-1)*2+ii 
              inod = (eny_j-2 +jj)*nx + enx_i + ii-1
              ndglno((ele-1)*nloc+iloc)=inod 
              x_all(1,(ele-1)*nloc+iloc) = real(enx_i-2+ii ) * dx
              x_all(2,(ele-1)*nloc+iloc) = real(eny_j-2+jj ) * dy 
              if(  (x_all(1,(ele-1)*nloc+iloc)>1.9) &
              .and.(x_all(1,(ele-1)*nloc+iloc)<3.9) ) then
                 if(  (x_all(2,(ele-1)*nloc+iloc)>1.9) &
                 .and.(x_all(2,(ele-1)*nloc+iloc)<3.9) ) then
                     t_old(inod)=1.0
                 endif
              endif
           end do
           end do
        end do
        end do
        t_new=t_old
!        print *,'t_new:',t_new
!        stop 22
!
        count=0
        do j=1,ny
        do i=1,nx
           inod=(j-1)*nx+i
           fina(inod)=count+1
           do jj=-1,1
           do ii=-1,1
              iii=i+ii
              jjj=j+jj
              if((iii>0).and.(jjj>0)) then 
              if((iii<nx+1).and.(jjj<ny+1)) then 
                 jnod=(jjj-1)*nx + iii
                 count=count+1
                 cola(count)=jnod
              endif
              endif
           end do
           end do
        end do
        end do
        fina(nx*ny+1)=count+1 
        end subroutine set_up_grid_problem
! 
! 
! 
! 
! in python:
! nlevel = calculate_nlevel_sfc(nonods)
        subroutine calculate_nlevel_sfc(nlevel,nonods)
! this subroutine calculates the number of multi-grid levels for a 
! space filling curve multi-grid or 1d multi-grid applied to nonods nodes. 
        implicit none
        integer, intent( in ) :: nonods
        integer, intent( out ) :: nlevel
! local variables...
        integer sfc_nonods_fine,sfc_nonods_course,ilevel
! coarsen...
        if(nonods==1) then
           nlevel=1 
        else
           sfc_nonods_course=nonods
           do ilevel=2,200
              sfc_nonods_fine=sfc_nonods_course
              sfc_nonods_course = (sfc_nonods_fine-1)/2 + 1
              nlevel=ilevel
              if(sfc_nonods_course==1) exit
           end do
        endif
        return
        end subroutine calculate_nlevel_sfc
! 
! in python:
! a_sfc, b_sfc, ml_sfc, fin_sfc_nonods, nonods_sfc_all_grids, nlevel = sfc_mapping_to_sfc_matrix( &
!                                     a, b, ml, &
!                                     fina,cola, ncola, sfc_node_ordering, &
!                                     nonods, max_nonods_sfc_all_grids, max_nlevel) 
!  
      subroutine best_sfc_mapping_to_sfc_matrix(a_sfc, b_sfc, ml_sfc, fin_sfc_nonods, nonods_sfc_all_grids, nlevel, &
                                     a, b, ml, &
                                     fina,cola, sfc_node_ordering, ncola, &
                                     nonods, max_nonods_sfc_all_grids, max_nlevel) 
! this subroutine finds the space filling curve representation of matrix eqns A T=b 
! - that is it forms matrix a and vector b and the soln vector is T 
! although T is not needed here. 
! It uses the BEST approach we can to form these tridigonal matrix approximations on different grids. 
! It also puts the vector b in space filling curve ordering. 
! it forms a series of matricies and vectors on a number of increasing coarse 1d grids 
! from nonods in length to 1 in length and stores this matrix in a_sfc. Similarly for the vectors b,ml 
! which are stored in b_sfc, ml_sfc. 
        implicit none
! fin_sfc_nonods(ilevel)=the start of course level ilevel and there 
! are nlevel grids from course to fine. 
! nonods_sfc_all_grids=total number of nodes all in all the grid levels. 
! ml is a vector possibly contsining the mass assocated with each cell/node of the original finite mesh. 
! sfc_node_ordering(i_sfc_order)=fem node number. Here i_sfc_order is the number of the node meansured along 
! the space filling curve trajectory. 
! nonods=number of finite element nodes in the mesh.
! max_nonods_sfc_all_grids = max number of nodes e.g. use 4*nonods.
! max_nlevel= max number of grid levels(e.g.=100). It can also be calculated from the subroutine  
! call in python: nlevel = calculate_nlevel_sfc(nonods)
! 
! fina,cola, ncola are used to define the sparcity pattern of the matrix. 
! ncola=number of potentially none-zeros in the nonods*nonods matrix a.  
! cola(count)=coln of the matrix a associated with entry count of matrix a - that is a(count).  
! fina(inod) start of the inod row of a matrix.
! SUPPOSE THE MESH COMPRISSES OF 2 RECTANGULAR ELEMENTS AS BELOW...
!      1-----2-----3
!      !     !     !
!      !     !     !        ndglno(1:4)=4,5,1,2                ndglno(5:8)=5,6,2,3
!      4-----5-----6
! THEN THE MATRIX HAS THE FORM (X is a non-zero entry in the matrix):
!           1  2  3  4  5  6 - column
! row 1    (X  X  0  X  X  0)
! row 2    (X  X  X  X  X  X)
! row 3    (0  X  X  0  X  X)
! row 4    (X  X  0  X  X  0)
! row 5    (X  X  X  X  X  X)
! row 6    (0  X  X  0  X  X)
! The comparact row storage only stores the non-zeros. 
! cola(1 )=1, cola(2 )=2, cola(3 )=4, cola(4 )=5,                                *****row 1   fina(1)=1
! cola(5 )=1, cola(6 )=2, cola(7 )=3, cola(8 )=4, cola(9 )=5, cola(10)=6,        *****row 2   fina(2)=5
! cola(11)=2, cola(12)=3, cola(13)=5, cola(14)=6,                                *****row 3   fina(3)=11
! cola(15)=1, cola(16)=2, cola(17)=4, cola(18)=5,                                *****row 4   fina(4)=15
! cola(19)=1, cola(20)=2, cola(21)=3, cola(22)=4, cola(23)=5, cola(24)=6,        *****row 5   fina(5)=19
! cola(25)=2, cola(26)=3, cola(27)=5, cola(28)=6                                 *****row 6   fina(6)=25
!                                                                                             fina(7)=29
! 
! 
        integer, intent( in ) :: ncola, nonods, max_nonods_sfc_all_grids, max_nlevel
        real, intent( out ) :: a_sfc(3,max_nonods_sfc_all_grids), b_sfc(max_nonods_sfc_all_grids), &
                               ml_sfc(max_nonods_sfc_all_grids) 
        real, intent( in ) :: a(ncola), b(nonods), ml(nonods)
        integer, intent( out ) :: nonods_sfc_all_grids, fin_sfc_nonods(max_nlevel), nlevel
        integer, intent( in ) :: fina(nonods+1), cola(ncola)
        integer, intent( in ) :: sfc_node_ordering(nonods)
! local variables...
        integer, allocatable :: sfc_node_ordering_inverse(:)
        integer i, count, nodj, nodi_sfc, ilevel, ilevel2
        integer ifinest_nod_sfc, jfinest_nod_sfc, icourse_nod_sfc, jcourse_nod_sfc, icourse_nod_sfc_displaced
        integer ifinest_nod, jfinest_nod
        integer sfc_nonods_fine, sfc_nonods_course, sfc_nonods_accum 
! 
! calculate nlevel from nonods
        call calculate_nlevel_sfc(nlevel,nonods)
! form SFC matrix...
        a_sfc(:,:)=0.0
        b_sfc(:)=0.0; b_sfc(1:nonods)=b(1:nonods)
        ml_sfc(:)=0.0; ml_sfc(1:nonods)=ml(1:nonods)
! 
! coarsen...
        sfc_nonods_accum=1
        fin_sfc_nonods(1)=sfc_nonods_accum
        sfc_nonods_accum=sfc_nonods_accum + nonods
        fin_sfc_nonods(2)=sfc_nonods_accum 
        do ilevel=2,nlevel
           sfc_nonods_fine=fin_sfc_nonods(ilevel)-fin_sfc_nonods(ilevel-1)
           if(sfc_nonods_fine.le.1) stop 13331 ! something went wrong. 
           sfc_nonods_course = (sfc_nonods_fine-1)/2 + 1
!           call map_sfc_course_grid( a_sfc(:,fin_sfc_nonods(ilevel)),sfc_nonods_course, &
!                                     a_sfc(:,fin_sfc_nonods(ilevel-1)),sfc_nonods_fine)

           call map_sfc_course_grid_vec( ml_sfc(fin_sfc_nonods(ilevel)),sfc_nonods_course, &
                                         ml_sfc(fin_sfc_nonods(ilevel-1)),sfc_nonods_fine)
           call map_sfc_course_grid_vec( b_sfc(fin_sfc_nonods(ilevel)),sfc_nonods_course, &
                                         b_sfc(fin_sfc_nonods(ilevel-1)),sfc_nonods_fine)
           sfc_nonods_accum = sfc_nonods_accum + sfc_nonods_course
           fin_sfc_nonods(ilevel+1)=sfc_nonods_accum
        end do
        nonods_sfc_all_grids=sfc_nonods_accum
        if(max_nonods_sfc_all_grids<nonods_sfc_all_grids) then
           print *,'run out of memory here stopping'
           stop 2822
        endif
!
       allocate(sfc_node_ordering_inverse(nonods))
       do ifinest_nod_sfc=1,nonods
          ifinest_nod = sfc_node_ordering(ifinest_nod_sfc) 
          sfc_node_ordering_inverse(ifinest_nod) = ifinest_nod_sfc
       end do 
! 
! coarsen...
        do ilevel=1,nlevel
           ilevel2=ilevel**2
           do ifinest_nod_sfc=1,nonods
              icourse_nod_sfc = 1 + (ifinest_nod_sfc-1)/ilevel2
              icourse_nod_sfc_displaced = icourse_nod_sfc + fin_sfc_nonods(ilevel) - 1
              ifinest_nod = sfc_node_ordering(ifinest_nod_sfc) 
              do count=fina(ifinest_nod),fina(ifinest_nod+1)-1
                 jfinest_nod=cola(count)
                 jfinest_nod_sfc = sfc_node_ordering_inverse(jfinest_nod)
                 jcourse_nod_sfc = 1 + (jfinest_nod_sfc-1)/ilevel2
                 if(jcourse_nod_sfc==icourse_nod_sfc-1) then
                    a_sfc(1,icourse_nod_sfc_displaced)=a_sfc(1,icourse_nod_sfc_displaced)+a(count)
                 else if(jcourse_nod_sfc==icourse_nod_sfc+1) then
                    a_sfc(3,icourse_nod_sfc_displaced)=a_sfc(3,icourse_nod_sfc_displaced)+a(count)
                 else 
                    a_sfc(2,icourse_nod_sfc_displaced)=a_sfc(2,icourse_nod_sfc_displaced)+a(count) ! diagonal
                 endif
              end do

           end do
        end do
        return 
        end subroutine best_sfc_mapping_to_sfc_matrix
! 
! 
! 
! in python:
! a_sfc, b_sfc, ml_sfc, fin_sfc_nonods, nonods_sfc_all_grids, nlevel = sfc_mapping_to_sfc_matrix( &
!                                     a, b, ml, &
!                                     fina,cola, ncola, sfc_node_ordering, &
!                                     nonods, max_nonods_sfc_all_grids, max_nlevel) 
!  
      subroutine sfc_mapping_to_sfc_matrix(a_sfc, b_sfc, ml_sfc, fin_sfc_nonods, nonods_sfc_all_grids, nlevel, &
                                     a, b, ml, &
                                     fina,cola, sfc_node_ordering, ncola, &
                                     nonods, max_nonods_sfc_all_grids, max_nlevel) 
! this subroutine finds the space filling curve representation of matrix eqns A T=b 
! - that is it forms matrix a and vector b and the soln vector is T 
! although T is not needed here. 
! It also puts the vector b in space filling curve ordering. 
! it forms a series of matricies and vectors on a number of increasing coarse 1d grids 
! from nonods in length to 1 in length and stores this matrix in a_sfc. Similarly for the vectors b,ml 
! which are stored in b_sfc, ml_sfc. 
        implicit none
! fin_sfc_nonods(ilevel)=the start of course level ilevel and there 
! are nlevel grids from course to fine. 
! nonods_sfc_all_grids=total number of nodes all in all the grid levels. 
! ml is a vector possibly contsining the mass assocated with each cell/node of the original finite mesh. 
! sfc_node_ordering(i_sfc_order)=fem node number. Here i_sfc_order is the number of the node meansured along 
! the space filling curve trajectory. 
! nonods=number of finite element nodes in the mesh.
! max_nonods_sfc_all_grids = max number of nodes e.g. use 4*nonods.
! max_nlevel= max number of grid levels(e.g.=100). It can also be calculated from the subroutine  
! call in python: nlevel = calculate_nlevel_sfc(nonods)
! 
! fina,cola, ncola are used to define the sparcity pattern of the matrix. 
! ncola=number of potentially none-zeros in the nonods*nonods matrix a.  
! cola(count)=coln of the matrix a associated with entry count of matrix a - that is a(count).  
! fina(inod) start of the inod row of a matrix.
! SUPPOSE THE MESH COMPRISSES OF 2 RECTANGULAR ELEMENTS AS BELOW...
!      1-----2-----3
!      !     !     !
!      !     !     !        ndglno(1:4)=4,5,1,2                ndglno(5:8)=5,6,2,3
!      4-----5-----6
! THEN THE MATRIX HAS THE FORM (X is a non-zero entry in the matrix):
!           1  2  3  4  5  6 - column
! row 1    (X  X  0  X  X  0)
! row 2    (X  X  X  X  X  X)
! row 3    (0  X  X  0  X  X)
! row 4    (X  X  0  X  X  0)
! row 5    (X  X  X  X  X  X)
! row 6    (0  X  X  0  X  X)
! The comparact row storage only stores the non-zeros. 
! cola(1 )=1, cola(2 )=2, cola(3 )=4, cola(4 )=5,                                *****row 1   fina(1)=1
! cola(5 )=1, cola(6 )=2, cola(7 )=3, cola(8 )=4, cola(9 )=5, cola(10)=6,        *****row 2   fina(2)=5
! cola(11)=2, cola(12)=3, cola(13)=5, cola(14)=6,                                *****row 3   fina(3)=11
! cola(15)=1, cola(16)=2, cola(17)=4, cola(18)=5,                                *****row 4   fina(4)=15
! cola(19)=1, cola(20)=2, cola(21)=3, cola(22)=4, cola(23)=5, cola(24)=6,        *****row 5   fina(5)=19
! cola(25)=2, cola(26)=3, cola(27)=5, cola(28)=6                                 *****row 6   fina(6)=25
!                                                                                             fina(7)=29
! 
! 
        integer, intent( in ) :: ncola, nonods, max_nonods_sfc_all_grids, max_nlevel
        real, intent( out ) :: a_sfc(3,max_nonods_sfc_all_grids), b_sfc(max_nonods_sfc_all_grids), &
                               ml_sfc(max_nonods_sfc_all_grids) 
        real, intent( in ) :: a(ncola), b(nonods), ml(nonods)
        integer, intent( out ) :: nonods_sfc_all_grids, fin_sfc_nonods(max_nlevel), nlevel
        integer, intent( in ) :: fina(nonods+1), cola(ncola)
        integer, intent( in ) :: sfc_node_ordering(nonods)
! local variables...
        integer i, count, nodj, prev_nodi_sfc, next_nodi_sfc, nodi_sfc, ilevel
        integer sfc_nonods_fine, sfc_nonods_course, sfc_nonods_accum 
! 
        print *,'dont use - use the best version' 
        stop 2231
! calculate nlevel from nonods
        call calculate_nlevel_sfc(nlevel,nonods)
! form SFC matrix...
        a_sfc(:,:)=0.0
        b_sfc(:)=0.0; b_sfc(1:nonods)=b(1:nonods)
        ml_sfc(:)=0.0; ml_sfc(1:nonods)=ml(1:nonods)
! 
        do i=1,nonods
           prev_nodi_sfc = 0
           next_nodi_sfc = 0
           if(i.ne.1) prev_nodi_sfc = sfc_node_ordering(i-1)
           if(i.ne.nonods) next_nodi_sfc = sfc_node_ordering(i+1)
           nodi_sfc=sfc_node_ordering(i)
           do count=fina(nodi_sfc),fina(nodi_sfc+1)-1
              nodj=cola(count)
              if(nodj==prev_nodi_sfc) then
                 a_sfc(1,nodi_sfc)=a_sfc(1,nodi_sfc)+a(count)
              else if(nodj==next_nodi_sfc) then
                 a_sfc(3,nodi_sfc)=a_sfc(3,nodi_sfc)+a(count)
              else 
                 a_sfc(2,nodi_sfc)=a_sfc(2,nodi_sfc)+a(count) ! diagonal
              endif
           end do
        end do
        sfc_nonods_accum=1
        fin_sfc_nonods(1)=sfc_nonods_accum
        sfc_nonods_accum=sfc_nonods_accum + nonods
        fin_sfc_nonods(2)=sfc_nonods_accum 
! 
! coarsen...
        do ilevel=2,nlevel
           sfc_nonods_fine=fin_sfc_nonods(ilevel)-fin_sfc_nonods(ilevel-1)
           if(sfc_nonods_fine.le.1) stop 13331 ! something went wrong. 
           sfc_nonods_course = (sfc_nonods_fine-1)/2 + 1
           call map_sfc_course_grid( a_sfc(:,fin_sfc_nonods(ilevel)),sfc_nonods_course, &
                                     a_sfc(:,fin_sfc_nonods(ilevel-1)),sfc_nonods_fine)

           call map_sfc_course_grid_vec( ml_sfc(fin_sfc_nonods(ilevel)),sfc_nonods_course, &
                                         ml_sfc(fin_sfc_nonods(ilevel-1)),sfc_nonods_fine)
           call map_sfc_course_grid_vec( b_sfc(fin_sfc_nonods(ilevel)),sfc_nonods_course, &
                                         b_sfc(fin_sfc_nonods(ilevel-1)),sfc_nonods_fine)
           sfc_nonods_accum = sfc_nonods_accum + sfc_nonods_course
           fin_sfc_nonods(ilevel+1)=sfc_nonods_accum
        end do
        nonods_sfc_all_grids=sfc_nonods_accum
        if(max_nonods_sfc_all_grids<nonods_sfc_all_grids) then
           print *,'run out of memory here stopping'
           stop 2822
        endif
        return 
        end subroutine sfc_mapping_to_sfc_matrix
! 
! 
! 
! 
        subroutine map_sfc_course_grid( a_sfc_course,sfc_nonods_course, a_sfc_fine,sfc_nonods_fine)
        implicit none 
        integer, intent( in ) :: sfc_nonods_course, sfc_nonods_fine
        real, intent( out ) :: a_sfc_course(3,sfc_nonods_course)
        real, intent( in ) :: a_sfc_fine(3,sfc_nonods_fine)
! local variables...
        integer i_short, i_long
! 
        a_sfc_course(:,:)=0.0
        do i_short=1,sfc_nonods_course
           i_long=(i_short-1)*2 + 1
           a_sfc_course(1,i_short)=a_sfc_course(1,i_short)+a_sfc_fine(1,i_long)
           a_sfc_course(2,i_short)=a_sfc_course(2,i_short)+a_sfc_fine(2,i_long)
           a_sfc_course(2,i_short)=a_sfc_course(2,i_short)+a_sfc_fine(3,i_long)

           i_long=(i_short-1)*2 + 2
           a_sfc_course(2,i_short)=a_sfc_course(2,i_short)+a_sfc_fine(1,i_long)
           a_sfc_course(2,i_short)=a_sfc_course(2,i_short)+a_sfc_fine(2,i_long)
           a_sfc_course(3,i_short)=a_sfc_course(3,i_short)+a_sfc_fine(3,i_long)
        end do
        
        return 
        end subroutine map_sfc_course_grid
! 
! 
! 
! 
        subroutine map_sfc_course_grid_vec( ml_sfc_course,sfc_nonods_course, ml_sfc_fine,sfc_nonods_fine)
        implicit none 
        integer, intent( in ) :: sfc_nonods_course, sfc_nonods_fine
        real, intent( out ) :: ml_sfc_course(sfc_nonods_course)
        real, intent( in ) :: ml_sfc_fine(sfc_nonods_fine)
! local variables...
        integer i_short, i_long
! 
        ml_sfc_course(:)=0.0
        do i_short=1,sfc_nonods_course
           i_long=(i_short-1)*2 + 1
           ml_sfc_course(i_short)=ml_sfc_course(i_short)+ml_sfc_fine(i_long)

           i_long=(i_short-1)*2 + 2
           ml_sfc_course(i_short)=ml_sfc_course(i_short)+ml_sfc_fine(i_long)
        end do
        
        return 
        end subroutine map_sfc_course_grid_vec
! 
! 
! 
! in python:
! t_new = time_step_filter_matrix( t_old, a_filter, fina,cola, ncola,nonods)
       subroutine time_step_filter_matrix(t_new, t_old, a_filter,   &
                                     fina,cola, ncola, nonods)
! this subroutine finds T^{n+1} = a_filter * T^n 
        implicit none
! nonods=number of finite element nodes in the mesh.
! dt = time step size. 
! 
! fina,cola, ncola are used to define the sparcity pattern of the matrix. 
! ncola=number of potentially none-zeros in the nonods*nonods matrix a.  
! cola(count)=coln of the matrix a associated with entry count of matrix a - that is a(count).  
! fina(inod) start of the inod row of a matrix.
! SUPPOSE THE MESH COMPRISSES OF 2 RECTANGULAR ELEMENTS AS BELOW...
!      1-----2-----3
!      !     !     !
!      !     !     !        ndglno(1:4)=4,5,1,2                ndglno(5:8)=5,6,2,3
!      4-----5-----6
! THEN THE MATRIX HAS THE FORM (X is a non-zero entry in the matrix):
!           1  2  3  4  5  6 - column
! row 1    (X  X  0  X  X  0)
! row 2    (X  X  X  X  X  X)
! row 3    (0  X  X  0  X  X)
! row 4    (X  X  0  X  X  0)
! row 5    (X  X  X  X  X  X)
! row 6    (0  X  X  0  X  X)
! The comparact row storage only stores the non-zeros. 
! cola(1 )=1, cola(2 )=2, cola(3 )=4, cola(4 )=5,                                *****row 1   fina(1)=1
! cola(5 )=1, cola(6 )=2, cola(7 )=3, cola(8 )=4, cola(9 )=5, cola(10)=6,        *****row 2   fina(2)=5
! cola(11)=2, cola(12)=3, cola(13)=5, cola(14)=6,                                *****row 3   fina(3)=11
! cola(15)=1, cola(16)=2, cola(17)=4, cola(18)=5,                                *****row 4   fina(4)=15
! cola(19)=1, cola(20)=2, cola(21)=3, cola(22)=4, cola(23)=5, cola(24)=6,        *****row 5   fina(5)=19
! cola(25)=2, cola(26)=3, cola(27)=5, cola(28)=6                                 *****row 6   fina(6)=25
!                                                                                             fina(7)=29
        integer, intent( in ) :: ncola, nonods
        real, intent( out ) :: t_new(nonods)
        real, intent( in ) :: a_filter(ncola)
        real, intent( in ) :: t_old(nonods)
        integer, intent( in ) :: fina(nonods+1), cola(ncola)
! local variables...  
        integer count,nodi,icol
! 
        t_new=0.0
        do nodi=1,nonods
           do count=fina(nodi),fina(nodi+1)-1
              icol=cola(count)
              t_new(nodi) = t_new(nodi) + a_filter(count) * t_old(icol) 
           end do
        end do
! 
        return
        end subroutine time_step_filter_matrix
! 
! in python:
! a_filter = get_filter_matrix( a, ml, dt, fina,cola, ncola,nonods)
       subroutine get_filter_matrix(a_filter, a, ml, dt,  &
                                     fina,cola, ncola,nonods)
! this subroutine finds the matrix eqns a_filter = -M_L^{-1} ( -M_L + dt* A). 
! Time stepping can be realised with T^{n+1} = a_filter * T^n 
        implicit none
! nonods=number of finite element nodes in the mesh.
! dt = time step size. 
! 
! fina,cola, ncola are used to define the sparcity pattern of the matrix. 
! ncola=number of potentially none-zeros in the nonods*nonods matrix a.  
! cola(count)=coln of the matrix a associated with entry count of matrix a - that is a(count).  
! fina(inod) start of the inod row of a matrix.
! SUPPOSE THE MESH COMPRISSES OF 2 RECTANGULAR ELEMENTS AS BELOW...
!      1-----2-----3
!      !     !     !
!      !     !     !        ndglno(1:4)=4,5,1,2                ndglno(5:8)=5,6,2,3
!      4-----5-----6
! THEN THE MATRIX HAS THE FORM (X is a non-zero entry in the matrix):
!           1  2  3  4  5  6 - column
! row 1    (X  X  0  X  X  0)
! row 2    (X  X  X  X  X  X)
! row 3    (0  X  X  0  X  X)
! row 4    (X  X  0  X  X  0)
! row 5    (X  X  X  X  X  X)
! row 6    (0  X  X  0  X  X)
! The comparact row storage only stores the non-zeros. 
! cola(1 )=1, cola(2 )=2, cola(3 )=4, cola(4 )=5,                                *****row 1   fina(1)=1
! cola(5 )=1, cola(6 )=2, cola(7 )=3, cola(8 )=4, cola(9 )=5, cola(10)=6,        *****row 2   fina(2)=5
! cola(11)=2, cola(12)=3, cola(13)=5, cola(14)=6,                                *****row 3   fina(3)=11
! cola(15)=1, cola(16)=2, cola(17)=4, cola(18)=5,                                *****row 4   fina(4)=15
! cola(19)=1, cola(20)=2, cola(21)=3, cola(22)=4, cola(23)=5, cola(24)=6,        *****row 5   fina(5)=19
! cola(25)=2, cola(26)=3, cola(27)=5, cola(28)=6                                 *****row 6   fina(6)=25
!                                                                                             fina(7)=29
        integer, intent( in ) :: ncola, nonods
        real, intent( out ) :: a_filter(ncola)
        real, intent( in ) :: a(ncola), ml(nonods), dt
        integer, intent( in ) :: fina(nonods+1), cola(ncola)
! local variables...  
        integer count,nodi,icol

        a_filter = dt*a 
        do nodi=1,nonods
           do count=fina(nodi),fina(nodi+1)-1
              icol=cola(count)
              if(icol==nodi) a_filter(count) = a_filter(count) - ml(nodi) 
           end do
        end do
! 
        do nodi=1,nonods
           do count=fina(nodi),fina(nodi+1)-1
              a_filter(count) = - a_filter(count)/ml(nodi) 
           end do
        end do
! 
        return
        end subroutine get_filter_matrix
! 
! 
! in python:
! a, b = u2r.get_fe_matrix_eqn(x_all, u, k, sig, s, fina,cola, ncola, ndglno, nonods,totele,nloc, ndim, ele_type)
! ml = the lumped mass 
        subroutine get_fe_matrix_eqn(a,b, ml, k, sig, s, u, x_all, &
                                     fina,cola, ndglno,  &
                                     ele_type, ndim, totele,nloc, ncola,nonods) 
! this subroutine finds the matrix eqns A T=b - that is it forms matrix a and vector b and the soln vector is T 
! although T is not needed here. 
        implicit none
! nonods=number of finite element nodes in the mesh.
! totele=no of elements in the mesh. 
! nloc=no of local nodes per element.
! ndim=no of dimensions.
! ele_type=element type index.
! 
! x_all contains the coordinates 
! and x_all(idim, (ele-1)*nloc+iloc) contains the idim'th coorindate for element number ele and local node 
! number iloc associated with element ele. 
! That is idim=1 corresponds to x coord, idim=2 y coordinate, idim=3 z coordinate.
!  
! u, k, sig, s the velocities, diffusion coefficient, absorption coefficient and source respectively of the differential equation. 
! The eqn we are solving is u\cdot\nabla T - \nabla \cdot k \nabla T + \sig T = s. 
! u(idim,inod) is the idim'th dimensional velocity (idim=1 corresponds to x-coord velocity) 
! of finite element node inod. 
! k(inod)=the diffusion coefficient of fem node nodi and similarly for sig, s. 
! 
! fina,cola, ncola are used to define the sparcity pattern of the matrix. 
! ncola=number of potentially none-zeros in the nonods*nonods matrix a.  
! cola(count)=coln of the matrix a associated with entry count of matrix a - that is a(count).  
! fina(inod) start of the inod row of a matrix.
! SUPPOSE THE MESH COMPRISSES OF 2 RECTANGULAR ELEMENTS AS BELOW...
!      1-----2-----3
!      !     !     !
!      !     !     !        ndglno(1:4)=4,5,1,2                ndglno(5:8)=5,6,2,3
!      4-----5-----6
! THEN THE MATRIX HAS THE FORM (X is a non-zero entry in the matrix):
!           1  2  3  4  5  6 - column
! row 1    (X  X  0  X  X  0)
! row 2    (X  X  X  X  X  X)
! row 3    (0  X  X  0  X  X)
! row 4    (X  X  0  X  X  0)
! row 5    (X  X  X  X  X  X)
! row 6    (0  X  X  0  X  X)
! The comparact row storage only stores the non-zeros. 
! cola(1 )=1, cola(2 )=2, cola(3 )=4, cola(4 )=5,                                *****row 1   fina(1)=1
! cola(5 )=1, cola(6 )=2, cola(7 )=3, cola(8 )=4, cola(9 )=5, cola(10)=6,        *****row 2   fina(2)=5
! cola(11)=2, cola(12)=3, cola(13)=5, cola(14)=6,                                *****row 3   fina(3)=11
! cola(15)=1, cola(16)=2, cola(17)=4, cola(18)=5,                                *****row 4   fina(4)=15
! cola(19)=1, cola(20)=2, cola(21)=3, cola(22)=4, cola(23)=5, cola(24)=6,        *****row 5   fina(5)=19
! cola(25)=2, cola(26)=3, cola(27)=5, cola(28)=6                                 *****row 6   fina(6)=25
!                                                                                             fina(7)=29
! 
! 
! ndglno contains the gloabal node number given a local node number and element number. 
! Thus ndglno((ele-1)*nloc+iloc) contains the global node number for element number ele and local node 
! number iloc associated with element ele.
!         
! local variables...
! 2d:
!        integer, parameter :: totele=1,nloc=4,sngi=3, ngi=9, ndim=2, nface=4, max_face_list_no=2
! 3d:
!        integer, parameter :: totele=1,nloc=8,sngi=9, ngi=27, ndim=3, nface=6, max_face_list_no=4
!        real :: n(ngi,nloc), nlx(ngi,ndim,nloc), weight(ngi)
!        real :: face_sn(sngi,nloc,nface), face_sn2(sngi,nloc,max_face_list_no), face_snlx(sngi,ndim,nloc,nface), face_sweigh(sngi,nface) 
        integer, intent( in ) :: ndim, totele, nloc, ele_type,  ncola, nonods
        real, intent( out ) :: a(ncola), b(nonods), ml(nonods)
        real, intent( in ) :: k(nonods), sig(nonods), s(nonods)
        real, intent( in ) :: u(ndim,nonods)
        real, intent( in ) :: x_all(ndim,nloc*totele)
        integer, intent( in ) :: fina(nonods+1), cola(ncola)
        integer, intent( in ) :: ndglno(nloc*totele)
! local variables...
        real, allocatable :: n(:,:), nlx(:,:,:), weight(:)
        real, allocatable :: face_sn(:,:,:), face_sn2(:,:,:)
        real, allocatable :: face_snlx(:,:,:,:), face_sweigh(:,:) 
        real, allocatable :: nx(:,:,:),detwei(:),inv_jac(:,:,:)
        real, allocatable :: x_loc(:,:)
        real, allocatable :: ugi(:,:), kgi(:), sgi(:), siggi(:)
        integer npoly,ele, iloc,jloc,idim
        integer ifil,jfil,kfil, iiblock, jjblock, kkblock, i,j
        integer count, nodi, nodj
        integer sngi, ngi, nface, max_face_list_no
        real nxnx_k, nnx_u, nnsig, aa 

! sngi=no of surface quadrature points of the faces - this is set to the max no of all faces.
! ngi=no of surface quadrature points of the faces. 
! nface=no of faces of each elemenet, nc no of fields to solve for.
        if(ele_type.ge.100) then ! triangle or tet...
           if(ndim==2) then
              sngi=2; ngi=3; nface=3; max_face_list_no=2
           else if(ndim==3) then
              sngi=3; ngi=4; nface=4; max_face_list_no=3
           endif
        else ! rectangle...
           if(ndim==2) then
              sngi=3; ngi=9; nface=4; max_face_list_no=2
           else if(ndim==3) then
              sngi=9; ngi=27; nface=6; max_face_list_no=4
           endif
        endif

! 
        allocate( n(ngi,nloc), nlx(ngi,ndim,nloc), weight(ngi) )
        allocate( face_sn(sngi,nloc,nface) )
        allocate( face_sn2(sngi,nloc,max_face_list_no) )
        allocate( face_snlx(sngi,ndim,nloc,nface), face_sweigh(sngi,nface) )

        allocate(nx(ngi,ndim,nloc),detwei(ngi),inv_jac(ngi,ndim,ndim))
!        allocate(nxnx(nloc,nloc),nnx(ndim,nloc,nloc))
        allocate(x_loc(ndim,nloc)) 

        allocate(ugi(ngi,ndim), kgi(ngi), sgi(ngi), siggi(ngi))

        npoly=1
!        ele_type=1

! form the shape functions...
        print *,'going into get_shape_funs_with_faces'
        call get_shape_funs_with_faces(n, nlx, weight,  &
               nloc, sngi, ngi, ndim, nface,max_face_list_no, &
               face_sn, face_sn2, face_snlx, face_sweigh, &
               npoly,ele_type) 
! 
!        print *,'k:',k
!        print *,'sig:',sig
!        print *,'s:',s 
!        print *,'u:',u
!        print *,'x_all:',x_all
!        print *,'fina:',fina
!        print *,'cola:',cola 
!        print *,'ndglno:',ndglno
!        print *,'ele_type, ndim, totele,nloc, ncola,nonods:', &
!                 ele_type, ndim, totele,nloc, ncola,nonods
!        print *,'n:',n
!        print *,'nlx:',nlx
! obtain 
! for filters...
        a=0.0
        b=0.0 
        ml=0.0
        do ele = 1, totele ! VOLUME integral
             x_loc(:,:) = x_all(:,(ele-1)*nloc+1:ele*nloc)
             call det_nlx( x_loc, n, nlx, nx, detwei, weight, ndim, nloc, ngi, INV_JAC )
!             print *,'nx, detwei, weight:',nx, detwei, weight
!
             ugi=0.0
             kgi=0.0
             sgi=0.0
             siggi=0.0
             do iloc=1,nloc
                nodi=ndglno((ele-1)*nloc+iloc) 
                do idim=1,ndim
                   ugi(:,idim) = ugi(:,idim) + n(:,iloc)*u(idim,nodi)
                end do
                kgi(:) = kgi(:) + n(:,iloc)*k(nodi)
                sgi(:) = sgi(:) + n(:,iloc)*s(nodi)
                siggi(:) = siggi(:) + n(:,iloc)*sig(nodi)
             end do

             do iloc=1,nloc
             nodi=ndglno((ele-1)*nloc+iloc) 
             b(nodi)=b(nodi)+sum( n(:,iloc)*detwei(:)*sgi(:) )
             ml(nodi)=ml(nodi)+sum( n(:,iloc)*detwei(:) )
             do jloc=1,nloc
                   nodj=ndglno((ele-1)*nloc+jloc) 
                   nxnx_k=0.0
                   nnx_u=0.0
                   do idim=1,ndim
                      nxnx_k = nxnx_k + sum( nx(:,idim,iloc)*kgi(:)*nx(:,idim,jloc)*detwei(:) )
                      nnx_u = nnx_u + sum( n(:,iloc)*ugi(:,idim)*nx(:,idim,jloc)*detwei(:) )
                   end do
                   nnsig = sum( n(:,iloc)*siggi(:)*n(:,jloc)*detwei(:) )
                   aa=nxnx_k + nnx_u + nnsig
!                   do idim=1,1
!                      nxnx(iloc,jloc) = nxnx(iloc,jloc) + sum( nx(:,idim,iloc)*nx(:,idim,jloc)*detwei(:) )
!                   end do
!                   print *,'ugi:',ugi
!                   print *,'aa, nxnx_k, nnx_u, nnsig:',aa, nxnx_k, nnx_u, nnsig
                   do count=fina(nodi),fina(nodi+1)-1
                      if(cola(count)==nodj) a(count) = a(count) + aa
                   end do
             end do
             end do
        end do
! 
        return
        end subroutine get_fe_matrix_eqn
! 


!subroutine det_nlx( x_loc, n, nlx, nx, detwei, weight, ndim, nloc, ngi, jac )
subroutine det_nlx( x_loc, n, nlx, nx, detwei, weight, ndim, nloc, ngi, INV_JAC )
  ! ****************************************************
  ! This sub form the derivatives of the shape functions
  ! ****************************************************
  ! x_loc: spatial nodes.
  ! n, nlx, nlx_lxx: shape function and local derivatives of the shape functions (nlx_lxx is local grad of the local laplacian)- defined by shape functional library.
  ! nx, nx_lxx: derivatives of the shape functions.
  ! detwei, inv_jac: determinant at the quadrature pots and inverse of Jacobian at quadrature pts.
  ! ndim,nloc,ngi: no of dimensions, no of local nodes within an element, no of quadrature points.
  ! nlx_nod, nx_nod: same as nlx and nx but formed at the nodes not quadrature points.
  implicit none
  integer, intent( in ) :: ndim,nloc,ngi

  REAL, DIMENSION( ndim,nloc ), intent( in ) :: x_loc
  REAL, DIMENSION( ngi, nloc ), intent( in ) :: N
  REAL, DIMENSION( ngi, ndim, nloc ), intent( in ) :: nlx
  REAL, DIMENSION( ngi ), intent( in ) :: WEIGHT
  REAL, DIMENSION( ngi ), intent( inout ) :: DETWEI
  REAL, DIMENSION( ngi, ndim, nloc ), intent( inout ) :: nx
  REAL, DIMENSION( ngi, ndim, ndim ), intent( inout ):: INV_JAC
  ! Local variables
  REAL :: AGI, BGI, CGI, DGI, EGI, FGI, GGI, HGI, KGI, A11, A12, A13, A21, &
          A22, A23, A31, A32, A33, DETJ
  INTEGER :: GI, L, IGLX, ii

  if (ndim==2) then
   ! conventional:
    do  GI=1,NGI! Was loop 331

      AGI=0.
      BGI=0.

      CGI=0.
      DGI=0.

      do  L=1,NLOC! Was loop 79
        IGLX=L

        AGI=AGI+NLX(GI,1,L)*x_loc(1,L)
        BGI=BGI+NLX(GI,1,L)*x_loc(2,L)

        CGI=CGI+NLX(GI,2,L)*x_loc(1,L)
        DGI=DGI+NLX(GI,2,L)*x_loc(2,L)

      end do ! Was loop 79

      DETJ= AGI*DGI-BGI*CGI
      DETWEI(GI)=ABS(DETJ)*WEIGHT(GI)
      ! For coefficient in the inverse mat of the jacobian.
      A11= DGI /DETJ
      A21=-BGI /DETJ

      A12=-CGI /DETJ
      A22= AGI /DETJ

      INV_JAC( GI, 1,1 )= A11
      INV_JAC( GI, 1,2 )= A21

      INV_JAC( GI, 2,1 )= A12
      INV_JAC( GI, 2,2 )= A22

      do  L=1,NLOC! Was loop 373
        NX(GI,1,L)= A11*NLX(GI,1,L)+A12*NLX(GI,2,L)
        NX(GI,2,L)= A21*NLX(GI,1,L)+A22*NLX(GI,2,L)
      end do ! Was loop 373
    end do ! GI Was loop 331

    !jac(1) = AGI; jac(2) = DGI ; jac(3) = BGI ; jac(4) = EGI

  elseif ( ndim.eq.3 ) then
    do  GI=1,NGI! Was loop 331

      AGI=0.
      BGI=0.
      CGI=0.

      DGI=0.
      EGI=0.
      FGI=0.

      GGI=0.
      HGI=0.
      KGI=0.

      do  L=1,NLOC! Was loop 79
        IGLX=L
        !ewrite(3,*)'xndgln, x, nl:', &
        !     iglx, l, x(iglx), y(iglx), z(iglx), NLX(L,GI), NLY(L,GI), NLZ(L,GI)
        ! NB R0 does not appear here although the z-coord might be Z+R0.
        AGI=AGI+NLX(GI,1,L)*x_loc(1,IGLX)
        BGI=BGI+NLX(GI,1,L)*x_loc(2,IGLX)
        CGI=CGI+NLX(GI,1,L)*x_loc(3,IGLX)

        DGI=DGI+NLX(GI,2,L)*x_loc(1,IGLX)
        EGI=EGI+NLX(GI,2,L)*x_loc(2,IGLX)
        FGI=FGI+NLX(GI,2,L)*x_loc(3,IGLX)

        GGI=GGI+NLX(GI,3,L)*x_loc(1,IGLX)
        HGI=HGI+NLX(GI,3,L)*x_loc(2,IGLX)
        KGI=KGI+NLX(GI,3,L)*x_loc(3,IGLX)
      end do ! Was loop 79

      DETJ=AGI*(EGI*KGI-FGI*HGI)&
          -BGI*(DGI*KGI-FGI*GGI)&
          +CGI*(DGI*HGI-EGI*GGI)
      DETWEI(GI)=ABS(DETJ)*WEIGHT(GI)
      ! ewrite(3,*)'gi, detj, weight(gi)', gi, detj, weight(gi)
      ! rsum = rsum + detj
      ! rsumabs = rsumabs + abs( detj )
      ! For coefficient in the inverse mat of the jacobian.
      A11= (EGI*KGI-FGI*HGI) /DETJ
      A21=-(DGI*KGI-FGI*GGI) /DETJ
      A31= (DGI*HGI-EGI*GGI) /DETJ

      A12=-(BGI*KGI-CGI*HGI) /DETJ
      A22= (AGI*KGI-CGI*GGI) /DETJ
      A32=-(AGI*HGI-BGI*GGI) /DETJ

      A13= (BGI*FGI-CGI*EGI) /DETJ
      A23=-(AGI*FGI-CGI*DGI) /DETJ
      A33= (AGI*EGI-BGI*DGI) /DETJ

      INV_JAC( GI, 1,1 )= A11
      INV_JAC( GI, 2,1 )= A21
      INV_JAC( GI, 3,1 )= A31
          !
      INV_JAC( GI, 1,2 )= A12
      INV_JAC( GI, 2,2 )= A22
      INV_JAC( GI, 3,2 )= A32
          !
      INV_JAC( GI, 1,3 )= A13
      INV_JAC( GI, 2,3 )= A23
      INV_JAC( GI, 3,3 )= A33

      do  L=1,NLOC! Was loop 373
        NX(GI,1,L)= A11*NLX(GI,1,L)+A12*NLX(GI,2,L)+A13*NLX(GI,3,L)
        NX(GI,2,L)= A21*NLX(GI,1,L)+A22*NLX(GI,2,L)+A23*NLX(GI,3,L)
        NX(GI,3,L)= A31*NLX(GI,1,L)+A32*NLX(GI,2,L)+A33*NLX(GI,3,L)
      end do ! Was loop 373
    end do ! GI Was loop 331
  end if
end subroutine det_nlx






         subroutine get_shape_funs_spec(n, nlx, nlx_lxx, nlxx, weight, nlx_nod, &
          nloc, sngi, ngi, ndim, nface,max_face_list_no, face_sn, face_sn2, face_snlx, face_sweigh, &
          npoly, ele_type ) 
        implicit none 
! nloc=no of local nodes per element
! ngi=no of quadrature points 
! ndim=no of dimensions. 
        integer, intent(in) :: nloc, sngi, ngi, ndim, nface, max_face_list_no
! shape functions....
! if .not.got_shape_funs then get the shape functions else assume we have them already
! n, nlx are the volume shape functions and their derivatives in local coordinates.
! weight are the weights of the quadrature points.
! nlx_nod are the derivatives of the local coordinates at the nods. 
! nlx_lxx = the 3rd order local derivatives at the nodes. 
! face info:
! face_ele(iface, ele) = given the face no iface and element no return the element next to 
! the surface or if negative return the negative of the surface element number between element ele and face iface.
! face_list_no(iface, ele) returns the possible origantation number which defines the numbering 
! of the non-zeros of the nabouting element.  
        real, intent(inout) :: n(ngi,nloc), nlx(ngi,ndim,nloc), nlxx(ngi,nloc), nlx_lxx(ngi,ndim,nloc), weight(ngi)
        real, intent(inout) :: nlx_nod(nloc,ndim,nloc)
        real, intent(inout) :: face_sn(sngi,nloc,nface), face_sn2(sngi,nloc,max_face_list_no)
        real, intent(inout) :: face_snlx(sngi,ndim,nloc,nface), face_sweigh(sngi,nface) 
! npoly=order of polynomial in Cartesian space; ele_type=type of element including order of poly. 
        integer, intent(in) :: npoly,ele_type

! form the shape functions...
        call get_shape_funs_with_faces(n, nlx, weight,  &
               nloc, sngi, ngi, ndim, nface,max_face_list_no, &
               face_sn, face_sn2, face_snlx, face_sweigh, &
               npoly,ele_type) 
          
! Calculate high order derivatives of shape functions nlx_lxx, nlxx 
! and also calculate node-wise deriavtives of shape functions nlx_nod. 
        call get_high_order_shape_funs(n, nlx, nlx_lxx, nlxx, &
               weight, nlx_nod, nloc, ngi, ndim, &
               npoly, ele_type) 

        end subroutine get_shape_funs_spec




        subroutine get_high_order_shape_funs(n, nlx, nlx_lxx, nlxx, &
               weight, nlx_nod, nloc, ngi, ndim, &
               npoly, ele_type) 
! ********************************************************************
! Calculate high order derivatives of shape functions nlx_lxx, nlxx 
! and also calculate node-wise deriavtives of shape functions nlx_nod. 
! ********************************************************************
        implicit none 
! nloc=no of local nodes per element
! ngi=no of quadrature points 
! ndim=no of dimensions. 
        integer, intent(in) :: nloc, ngi, ndim, npoly, ele_type
! shape functions....
! if .not.got_shape_funs then get the shape functions else assume we have them already
! n, nlx are the volume shape functions and their derivatives in local coordinates.
! weight are the weights of the quadrature points.
! nlx_nod are the derivatives of the local coordinates at the nods. 
! nlx_lxx = the 3rd order local derivatives in local coords.
! nlxx = the 2rd order local derivatives or Laplacian in local coords. 
        real, intent(in) :: n(ngi,nloc), nlx(ngi,ndim,nloc), weight(ngi)
        real, intent(inout) :: nlx_lxx(ngi,ndim,nloc),  nlxx(ngi,nloc)
        real, intent(inout) :: nlx_nod(nloc,ndim,nloc)
! local variables...
        integer iloc,jloc,idim
        real, allocatable :: nn(:,:), nn_inv(:,:), nnlx(:,:,:) 
        real, allocatable :: mat(:,:),mat2(:,:), x(:),b(:) 
        real, allocatable :: vec(:),sol(:),rhs_nod_lxx(:)
        
        allocate(nn(nloc,nloc), nn_inv(nloc,nloc), nnlx(nloc,nloc,ndim) )
        allocate(mat(nloc,nloc),mat2(nloc,nloc), x(nloc),b(nloc) )
        allocate(vec(nloc),sol(nloc),rhs_nod_lxx(nloc))

! Calculate high order derivatives of shape functions nlx_lxx, nlxx 
! and also calculate node-wise deriavtives of shape functions nlx_nod: 

! Calculate nodal values of nlx, that is nlx_nod:
! form nn
        do iloc=1,nloc
           do jloc=1,nloc
              nn(iloc,jloc)= sum( n(:,iloc)*n(:,jloc)*weight(:) )
              do idim=1,ndim
                 nnlx(iloc,jloc,idim)= sum( n(:,iloc)*nlx(:,idim,jloc)*weight(:) )
              end do
           end do
        end do
! find nn_inv as its easy to manipulate then although not efficient.
        nn_inv=nn
        call matinv(nn_inv,nloc,nloc,MAT,MAT2,X,B)

! Form nlx_nod
        do iloc=1,nloc
        do idim=1,ndim
           vec(:) = nnlx(iloc,:,idim)
           nlx_nod(iloc,idim,:) = matmul( nn_inv(:,:), vec(:) )
        end do
        end do

! Form nlxx from nlx_nod...
        nlxx=0.0
        do iloc=1,nloc
           do idim=1,ndim
              vec(:) = sum( nnlx(iloc,:,idim)*nlx_nod(iloc,idim,:) )
              sol(:) = matmul( nn_inv(:,:), vec(:) )
              nlxx(iloc,:) = nlxx(iloc,:) +sol(:) 
           end do
        end do

! Form nlx_lxx: 
        do iloc=1,nloc
           rhs_nod_lxx(:)=nlxx(iloc,:)
           do idim=1,ndim
              nlx_lxx(:,idim,iloc) = nlx(:,idim,iloc)*rhs_nod_lxx(:) 
           end do
        end do
          
        end subroutine get_high_order_shape_funs

     



        subroutine get_shape_funs_with_faces(n, nlx, weight,  &
               nloc, sngi, ngi, ndim, nface,max_face_list_no, &
               face_sn, face_sn2, face_snlx, face_sweigh, &
               npoly,ele_type) 
        implicit none 
! nloc=no of local nodes per element
! ngi=no of quadrature points 
! ndim=no of dimensions. 
! ele_type= element type
        integer, intent(in) :: nloc, sngi, ngi, ndim, nface, max_face_list_no
! shape functions....
! if .not.got_shape_funs then get the shape functions else assume we have them already
! n, nlx are the volume shape functions and their derivatives in local coordinates.
! weight are the weights of the quadrature points.
! nlx_nod are the derivatives of the local coordinates at the nods. 
! nlx_lxx = the 3rd order local derivatives at the nodes. 
! face info:
! face_ele(iface, ele) = given the face no iface and element no return the element next to 
! the surface or if negative return the negative of the surface element number between element ele and face iface.
! face_list_no(iface, ele) returns the possible origantation number which defines the numbering 
! of the non-zeros of the nabouting element.  
        real, intent(inout) :: n(ngi,nloc), nlx(ngi,ndim,nloc), weight(ngi)
        real, intent(inout) :: face_sn(sngi,nloc,nface)
        real, intent(inout) :: face_sn2(sngi,nloc,max_face_list_no)
        real, intent(inout) :: face_snlx(sngi,ndim,nloc,nface)
        real, intent(inout) :: face_sweigh(sngi,nface) 
! npoly=order of polynomial in Cartesian space; ele_type=type of element including order of poly. 
        integer, intent(in) :: npoly, ele_type
! local variables...
        integer is_triangle_or_tet
        parameter(is_triangle_or_tet=100) 
        integer sndim, suf_ngi, suf_ndim, suf_nloc, ipoly, IQADRA
        real, allocatable :: rdum1(:), rdum2(:), rdum3(:) 
        real, allocatable :: sn(:,:),sn2(:,:),snlx(:,:,:),sweight(:) 
        real, allocatable :: suf_n(:,:),suf_nlx(:,:,:),suf_weight(:)

! allocate memory...
        print *,'just about to allocate'
        print *,'nloc, sngi, ngi, ndim, nface,max_face_list_no:', &
                 nloc, sngi, ngi, ndim, nface,max_face_list_no
        print *,'npoly,ele_type:',npoly,ele_type
        allocate(rdum1(10000), rdum2(10000), rdum3(10000) )
        print *,'here 1'
        allocate(sn(sngi,nloc),sn2(sngi,nloc),snlx(sngi,ndim,nloc),sweight(sngi) )
        print *,'here 2'
        allocate(suf_n(sngi,nloc),suf_nlx(sngi,ndim,nloc),suf_weight(sngi) )
        print *,'here 3'

        ipoly=npoly
!        IQADRA=IPOLY+1
        IQADRA=1

        sndim=ndim-1
        if(ele_type < is_triangle_or_tet) then ! not triangle...

           print *,'get_shape_funs'
           call get_shape_funs(ngi,nloc,ndim,  &
             weight,n,nlx, ipoly,iqadra, &
             sngi, sndim, sweight,sn,snlx, .true.   )
           print *,'finished get_shape_funs'

        else 
!           allocate(rdum1(10000),rdum2(10000),rdum3(10000))
! gives a surface triangle with time slab in surface integral. 
           print *,'going into get_shape_funs ***is a triangle or tet'
           call get_shape_funs(ngi,nloc,ndim,  &
              weight,n,nlx, ipoly,iqadra, &
              sngi, sndim, sweight,sn,snlx, .false.   )
           print *,'out of get_shape_funs'
! return a surface tet...
           suf_ngi=NGI/IQADRA
           suf_ndim=ndim-1
           suf_nloc=NLOC/(IPOLY+1)
           call get_shape_funs(suf_ngi,suf_nloc,suf_ndim,  &
              suf_weight,suf_n,suf_nlx, ipoly,iqadra, &
              sngi, sndim, rdum1,rdum2,rdum3, .false.   )

        endif

        end subroutine get_shape_funs_with_faces
     



       subroutine get_shape_funs(ngi,nloc,ndim,  &
             weight,n,nlx, ipoly,iqadra, &
             sngi, sndim, sweight,sn,snlx, with_time_slab   )
! ***************************************
! form volume and surface shape functions
! ***************************************
    IMPLICIT NONE
    INTEGER, intent(in) :: sngi, NGI, NLOC, ndim, sndim
    INTEGER, intent(in) :: IPOLY,IQADRA
    logical, intent(in) :: with_time_slab
    REAL, intent(inout) ::  n(ngi,nloc) 
    REAL, intent(inout) ::  nlx(ngi,ndim,nloc)
    REAL, intent(inout) ::  sn(sngi, nloc) 
    REAL, intent(inout) ::  snlx(sngi, sndim, nloc)  
    real, intent(inout) :: WEIGHT(ngi), sWEIGHT(sngi)
! local variables...
    integer mloc, NDNOD, snloc_temp
    logical square
    real, allocatable :: m(:)

    allocate(m(10000)) 
    mloc=1
    SQUARE = .true. 
    if((nloc==3).and.(ndim==2)) then
       SQUARE = .false. 
    endif
    if((nloc==4).and.(ndim==3)) then
       SQUARE = .false. 
    endif

    IF(SQUARE) THEN ! Square in up to 4D
! volumes
!       print *,'going into interface_SPECTR NGI,NLOC,ndim:',NGI,NLOC,ndim
       call interface_SPECTR(NGI,NLOC,WEIGHT,N,NLX, ndim, IPOLY,IQADRA  )
!       print *,'out of interface_SPECTR'
! surfaces...
       NDNOD =INT((NLOC**(1./real(ndim) ))+0.1)
       snloc_temp=NDNOD**(ndim-1) 
       sWEIGHT=0.0; sN=0.0; sNLX=0.0
!       print *,'surfaces going into interface_SPECTR'
       call interface_SPECTR(sNGI,sNLOC_temp,sWEIGHT,sN,sNLX, ndim-1, IPOLY,IQADRA  )
!       print *,'surfaces out of interface_SPECTR'
       
    ELSE
! volume tet plus time slab...
!       print *,'triangles or tets ipoly,iqadra, with_time_slab:',ipoly,iqadra, with_time_slab
       call triangles_tets_with_time( nloc,ngi, ndim, &
               n,nlx, weight, ipoly,iqadra, with_time_slab) 
! surfaces...
!       print *,'surfaces -tets or triangles...'
       call triangles_tets_with_time( nloc,sngi, ndim-1, &
              sn,snlx, sweight, ipoly,iqadra, with_time_slab) 
    ENDIF

     end subroutine get_shape_funs




     subroutine triangles_tets_with_time( nloc,ngi, ndim, &
                  n,nlx, weight, ipoly,iqadra, with_time_slab) 
! ****************************************************************************************
     implicit none
     integer, intent(in) :: nloc,ngi,ndim, ipoly,iqadra
     real, intent(inout) :: n(ngi,nloc), nlx(ngi,ndim,nloc) 
     real, intent(inout) :: weight(ngi)
     logical, intent(in) :: with_time_slab
! local variables...
     integer nloc_space, ngi_space, ndim_space,   nloc_t, ngi_t, ndim_t
     logical d3
     real, allocatable :: l1(:), l2(:), l3(:), l4(:)
     real, allocatable :: weight_space(:), n_space(:,:), nlx_space(:,:,:)
     real, allocatable :: weight_t(:), n_t(:,:), nlx_t(:,:,:)

     if(with_time_slab) then ! form a space-time slab.

        nloc_t=nloc/(ipoly+1)
        NGI_T=IQADRA
!        ngi_t=ngi/(ipoly+2)
        ndim_t = 1

        nloc_space=nloc/nloc_t 
        ngi_space=ngi/ngi_t
        ndim_space = ndim-1
        allocate(l1(ngi_space), l2(ngi_space), l3(ngi_space), l4(ngi_space) )
        allocate(weight_space(ngi_space), n_space(ngi_space,nloc_space), &
                 nlx_space(ngi_space,ndim_space,nloc_space) )

        allocate(weight_t(ngi_t), n_t(ngi_t,nloc_t), nlx_t(ngi_t,ndim_t,nloc_t) )

! triangles or tetrahedra...
        call SHATRInew(L1, L2, L3, L4, WEIGHT_space, &
          NLOC_space,NGI_space,ndim_space,  n_space,nlx_space)

! extend into time domain...
        call interface_SPECTR(NGI_t,NLOC_t,WEIGHT_t,N_t,NLX_t, &
                              ndim_t, IPOLY,IQADRA  )

! combine-space time...
        call make_space_time_shape_funs(nloc_space,ngi_space, ndim_space, &
                                     n_space,nlx_space, weight_space, &
                                     nloc_t,ngi_t, ndim_t, n_t, nlx_t, weight_t, &
                                     nloc,ngi, ndim, n, nlx, weight ) 
     else ! just a triangle or tet without time slab...
! triangles or tetrahedra...
        allocate(l1(ngi), l2(ngi), l3(ngi), l4(ngi) ) 
!        stop 22221
  !
        d3=(ndim==3)
        call TRIQUAold(L1, L2, L3, L4, WEIGHT, D3,NGI)

!        print *,'d3,nloc,ngi,ndim:',d3,nloc,ngi,ndim
!        print *,'l1,l2,l3:',l1,l2,l3
!        print *,'weight:',weight
        call SHATRInew(L1, L2, L3, L4, weight, &
          nloc,ngi,ndim,  n,nlx)
!        print *,'n:',n
!        stop 282
     endif

     end subroutine triangles_tets_with_time
  ! 



     subroutine make_space_time_shape_funs(nloc_space,ngi_space, ndim_space, &
                                           n_space,nlx_space, weight_space, &
                                           nloc_t,ngi_t, ndim_t, n_t, nlx_t, weight_t, &
                                           nloc,ngi, ndim, n, nlx, weight ) 
! ****************************************************************************************
! this sub convolves the space and time shape functions to get space-time shape functions. 
! these are returned in n, nlx where nloc=no of local nodes and ngi =no of quadrature pts.
! ****************************************************************************************
     implicit none
     integer, intent(in) :: nloc_space,nloc_t, ngi_space,ngi_t, nloc,ngi, &
                            ndim_space,ndim_t, ndim
     real, intent(in) :: n_space(ngi_space,nloc_space),n_t(ngi_t,nloc_t)
     real, intent(in) :: nlx_space(ngi_space,ndim_space,nloc_space), &
                         nlx_t(ngi_t,ndim_t,nloc_t)
     real, intent(in) :: weight_space(ngi_space), weight_t(ngi_t)
     real, intent(inout) :: n(ngi,nloc), nlx(ngi,ndim,nloc) 
     real, intent(inout) :: weight(ngi)
! local...
     integer iloc_space, iloc_t, iloc, gi_space, gi_t, gi, &
             idim_space, idim_t, idim 
     
! remember sngi>=the needed sngi for a surface element.
! similarly for nloc
     n=0.0
     nlx=0.0 
     weight=0.0
     do iloc_space=1,nloc_space
        do iloc_t=1,nloc_t
           iloc=(iloc_t-1)*nloc_space + iloc_space
           do gi_space=1,ngi_space
              do gi_t=1,ngi_t
                 gi=(gi_t-1)*ngi_space + gi_space
                 n(gi,iloc) = n_space(gi_space,iloc_space)*n_t(gi_t,iloc_t) 
                 do idim_space=1,ndim_space
                    nlx(gi,idim_space,iloc) = nlx_space(gi_space,idim_space,iloc_space)*n_t(gi_t,iloc_t)
                 end do
                 idim = ndim_space+1
                 nlx(gi,idim,iloc) = n(gi_space,iloc_space)*nlx_t(gi_t,1,iloc_t)
     end do; end do; end do; end do
! the weights...
     do gi_space=1,ngi_space
        do gi_t=1,ngi_t
           gi=(gi_t-1)*ngi_space + gi_space
           weight(gi) = weight_space(gi_space)*weight_t(gi_t)
        end do
     end do
     end subroutine make_space_time_shape_funs
  ! 


  ! 
       subroutine interface_SPECTR(NGI,NLOC,WEIGHT,N,NLX, ndim, IPOLY,IQADRA  )
    IMPLICIT NONE
    INTEGER , intent(in) :: NGI, NLOC, ndim
    INTEGER , intent(in) :: IPOLY,IQADRA
    REAL, dimension(ngi,nloc), intent(inout) ::  N
    REAL, dimension(ngi,ndim,nloc), intent(inout) ::  NLX   
    real, dimension(ngi), intent(inout) :: WEIGHT
! local variables...
    logical d2,d3,d4
    integer mloc,idim
    real, allocatable :: m(:),NLX_TEMP(:,:,:)

    allocate(m(10000),NLX_TEMP(NGI,4,nloc)) 
    mloc=nloc ! 1
    d2=.false.; d3=.false.; d4=.false.
    if(ndim==2) d2=.true.
    if(ndim==3) d3=.true.
    if(ndim==4) d4=.true.

!    IF(SQUARE) THEN ! Square in up to 4D
! volumes
       print *,'going into SPECTR'
       call SPECTR(NGI,NLOC,MLOC, &
       &      M,WEIGHT,N,NLX_temp(:,1,:),NLX_temp(:,2,:), &
              NLX_temp(:,3,:),NLX_temp(:,4,:),D4,D3,D2, IPOLY,IQADRA  )
       print *,'out of SPECTR'

    do idim=1,ndim
       nlx(:,idim,:) = nlx_temp(:,idim,:) 
    end do

    end subroutine interface_SPECTR




  SUBROUTINE SPECTR(NGI,NLOC,MLOC, &
              M,WEIGHT,N,NLX, & 
              NLY,NLZ,NLT,D4,D3,D2, IPOLY,IQADRA  )
    IMPLICIT NONE
    INTEGER, intent(in) :: NGI, NLOC, MLOC
    INTEGER, intent(in) :: IPOLY,IQADRA
    REAL, dimension(ngi,nloc), intent(inout) ::  M, N, NLX, NLY, NLZ, NLT
    real, dimension(ngi), intent(inout) :: WEIGHT
    logical, intent(in) :: d2,d3,d4
    !Local variables
    REAL :: RGPTWE
    REAL , dimension (300) :: WEIT,NODPOS,QUAPOS
    INTEGER :: GPOI
    LOGICAL :: DIFF,NDIFF
    INTEGER :: NDGI,NDNOD,NMDNOD,IGR,IGQ,IGP,KNOD,JNOD,INOD,ILOC,lnod,igs
    REAL :: LXGP,LYGP,LZGP,LTGP
! SPECFU is a real function
    real :: SPECFU
    ! This subroutine defines a spectal element.
    ! IPOLY defines the element type and IQADRA the quadrature.
    ! In 2-D the spectral local node numbering is as..
    ! 7 8 9
    ! 4 5 6
    ! 1 2 3
    ! For 3-D...
    ! lz=-1
    ! 3 4
    ! 1 2
    ! and for lz=1
    ! 7 8
    ! 5 6

    !ewrite(3,*)'inside SPECTR IPOLY,IQADRA', IPOLY,IQADRA
    !
    DIFF=.TRUE.
    NDIFF=.FALSE.
    IF(D4) THEN
       NDGI  =INT((NGI**(1./4.))+0.1)
       NDNOD =INT((NLOC**(1./4.))+0.1)
       NMDNOD=INT((MLOC**(1./4.))+0.1)
       !
       ! Find the roots of the quadrature points and nodes
       ! also get the weights.
       !ewrite(3,*)'about to go into inside GTROOT IPOLY,IQADRA',IPOLY,IQADRA
       CALL GTROOT(IPOLY,IQADRA,WEIT,NODPOS,QUAPOS,NDGI,NDNOD)
       !ewrite(3,*)'outside GTROOT'
       do  IGS=1,NDGI! Was loop 101
       do  IGR=1,NDGI! Was loop 101
          do  IGQ=1,NDGI! Was loop 101
             do  IGP=1,NDGI! Was loop 101
                GPOI=IGP + (IGQ-1)*NDGI + (IGR-1)*NDGI*NDGI + (IGS-1)*NDGI*NDGI*NDGI
                !
                !           WEIGHT(GPOI)
                !     &        =RGPTWE(IGP,NDGI,.TRUE.)*RGPTWE(IGQ,NDGI,.TRUE.)
                !     &        *RGPTWE(IGR,NDGI,.TRUE.)
                WEIGHT(GPOI)=WEIT(IGP)*WEIT(IGQ)*WEIT(IGR)*WEIT(IGS)
                !
                LXGP=QUAPOS(IGP)
                LYGP=QUAPOS(IGQ)
                LZGP=QUAPOS(IGR)
                LTGP=QUAPOS(IGS)
                ! NB If TRUE in function RGPTWE then return the Gauss-pt weight
                ! else return the Gauss-pt.
                !
                do  LNOD=1,NDNOD! Was loop 20
                do  KNOD=1,NDNOD! Was loop 20
                   do  JNOD=1,NDNOD! Was loop 20
                      do  INOD=1,NDNOD! Was loop 20
                         ILOC=INOD + (JNOD-1)*NDNOD + (KNOD-1)*NDNOD*NDNOD + &
                             (LNOD-1)*NDNOD*NDNOD*NDNOD
                         !
                         N(GPOI,ILOC) = &
                              SPECFU(NDIFF,LXGP,INOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(NDIFF,LYGP,JNOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(NDIFF,LZGP,KNOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(NDIFF,LTGP,KNOD,NDNOD,IPOLY,NODPOS)
                         !
                         NLX(GPOI,ILOC) = &
                              SPECFU(DIFF,LXGP, INOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(NDIFF,LYGP,JNOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(NDIFF,LZGP,KNOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(NDIFF,LTGP,KNOD,NDNOD,IPOLY,NODPOS)
                         !
                         NLY(GPOI,ILOC) = &
                              SPECFU(NDIFF,LXGP,INOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(DIFF, LYGP,JNOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(NDIFF,LZGP,KNOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(NDIFF,LTGP,KNOD,NDNOD,IPOLY,NODPOS)
                         !
                         NLZ(GPOI,ILOC) = &
                              SPECFU(NDIFF,LXGP,INOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(NDIFF,LYGP,JNOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(DIFF,LZGP, KNOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(DIFF,LTGP, KNOD,NDNOD,IPOLY,NODPOS)
                         !
                         NLT(GPOI,ILOC) = &
                              SPECFU(NDIFF,LXGP,INOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(NDIFF,LYGP,JNOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(DIFF,LZGP, KNOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(DIFF,LtGP, KNOD,NDNOD,IPOLY,NODPOS)
                         !
                      end do ! Was loop 20
                   end do ! Was loop 20
                end do ! Was loop 20
                end do ! Was loop 20
             end do ! Was loop 101
          end do ! Was loop 101
       end do ! Was loop 101
       end do ! Was loop 101
       !
       !
       ! Find the roots of the quadrature points and nodes
       ! also get the weights.
       !ewrite(3,*)'2about to go into inside GTROOT IPOLY,IQADRA',IPOLY,IQADRA
       CALL GTROOT(IPOLY,IQADRA,WEIT,NODPOS,QUAPOS,NDGI,NMDNOD)
       !ewrite(3,*)'2out of GTROOT'

       do  IGS=1,NDGI! Was loop 102
       do  IGR=1,NDGI! Was loop 102
          do  IGQ=1,NDGI! Was loop 102
             do  IGP=1,NDGI! Was loop 102
                GPOI=IGP + (IGQ-1)*NDGI + (IGR-1)*NDGI*NDGI &
                    + (IGS-1)*NDGI*NDGI*NDGI
                !
                LXGP=QUAPOS(IGP)
                LYGP=QUAPOS(IGQ)
                LZGP=QUAPOS(IGR)
                LTGP=QUAPOS(IGS)

                do  LNOD=1,NMDNOD! Was loop 30
                do  KNOD=1,NMDNOD! Was loop 30
                   do  JNOD=1,NMDNOD! Was loop 30
                      do  INOD=1,NMDNOD! Was loop 30
                         ILOC=INOD + (JNOD-1)*NMDNOD + (KNOD-1)*NMDNOD*NMDNOD &
                             + (LNOD-1)*NMDNOD*NMDNOD*NMDNOD
                         !
                         M(GPOI,ILOC) = &
                              SPECFU(NDIFF,LXGP,INOD,NMDNOD,IPOLY,NODPOS)  &
                              *SPECFU(NDIFF,LYGP,JNOD,NMDNOD,IPOLY,NODPOS) &
                              *SPECFU(NDIFF,LZGP,KNOD,NMDNOD,IPOLY,NODPOS)&
                              *SPECFU(NDIFF,LtGP,LNOD,NMDNOD,IPOLY,NODPOS)
                         !
                      end do ! Was loop 30
                   end do ! Was loop 30
                end do ! Was loop 30
                end do ! Was loop 30
             end do ! Was loop 102
          end do ! Was loop 102
       end do ! Was loop 102
       end do ! Was loop 102
    ENDIF ! ENDIF IF(D4) THEN

    IF(D3) THEN
       NDGI  =INT((NGI**(1./3.))+0.1)
       NDNOD =INT((NLOC**(1./3.))+0.1)
       NMDNOD=INT((MLOC**(1./3.))+0.1)
       !
       ! Find the roots of the quadrature points and nodes
       ! also get the weights.
       !ewrite(3,*)'about to go into inside GTROOT IPOLY,IQADRA',IPOLY,IQADRA
       CALL GTROOT(IPOLY,IQADRA,WEIT,NODPOS,QUAPOS,NDGI,NDNOD)
       !ewrite(3,*)'outside GTROOT'
       do  IGR=1,NDGI! Was loop 101
          do  IGQ=1,NDGI! Was loop 101
             do  IGP=1,NDGI! Was loop 101
                GPOI=IGP + (IGQ-1)*NDGI + (IGR-1)*NDGI*NDGI
                !
                !           WEIGHT(GPOI)
                !     &        =RGPTWE(IGP,NDGI,.TRUE.)*RGPTWE(IGQ,NDGI,.TRUE.)
                !     &        *RGPTWE(IGR,NDGI,.TRUE.)
                WEIGHT(GPOI)=WEIT(IGP)*WEIT(IGQ)*WEIT(IGR)
                !
                LXGP=QUAPOS(IGP)
                LYGP=QUAPOS(IGQ)
                LZGP=QUAPOS(IGR)
                ! NB If TRUE in function RGPTWE then return the Gauss-pt weight
                ! else return the Gauss-pt.
                !
                do  KNOD=1,NDNOD! Was loop 20
                   do  JNOD=1,NDNOD! Was loop 20
                      do  INOD=1,NDNOD! Was loop 20
                         ILOC=INOD + (JNOD-1)*NDNOD + (KNOD-1)*NDNOD*NDNOD
                         !
                         N(GPOI,ILOC) = &
                              SPECFU(NDIFF,LXGP,INOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(NDIFF,LYGP,JNOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(NDIFF,LZGP,KNOD,NDNOD,IPOLY,NODPOS)
!            print *,'GPOI,iloc,N(GPOI,ILOC):',GPOI,iloc,N(GPOI,ILOC)
                         !
                         NLX(GPOI,ILOC) = &
                              SPECFU(DIFF,LXGP, INOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(NDIFF,LYGP,JNOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(NDIFF,LZGP,KNOD,NDNOD,IPOLY,NODPOS)
!            print *,'GPOI,iloc,NLX(GPOI,ILOC):',GPOI,iloc,NLX(GPOI,ILOC)
                         !
                         NLY(GPOI,ILOC) = &
                              SPECFU(NDIFF,LXGP,INOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(DIFF, LYGP,JNOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(NDIFF,LZGP,KNOD,NDNOD,IPOLY,NODPOS)
                         !
                         NLZ(GPOI,ILOC) = &
                              SPECFU(NDIFF,LXGP,INOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(NDIFF,LYGP,JNOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(DIFF,LZGP, KNOD,NDNOD,IPOLY,NODPOS)
                         !
                      end do ! Was loop 20
                   end do ! Was loop 20
                end do ! Was loop 20
             end do ! Was loop 101
          end do ! Was loop 101
       end do ! Was loop 101
       !
       !
       ! Find the roots of the quadrature points and nodes
       ! also get the weights.
       !ewrite(3,*)'2about to go into inside GTROOT IPOLY,IQADRA',IPOLY,IQADRA
       CALL GTROOT(IPOLY,IQADRA,WEIT,NODPOS,QUAPOS,NDGI,NMDNOD)
       !ewrite(3,*)'2out of GTROOT'
       do  IGR=1,NDGI! Was loop 102
          do  IGQ=1,NDGI! Was loop 102
             do  IGP=1,NDGI! Was loop 102
                GPOI=IGP + (IGQ-1)*NDGI + (IGR-1)*NDGI*NDGI
                !
                LXGP=QUAPOS(IGP)
                LYGP=QUAPOS(IGQ)
                LZGP=QUAPOS(IGR)

                do  KNOD=1,NMDNOD! Was loop 30
                   do  JNOD=1,NMDNOD! Was loop 30
                      do  INOD=1,NMDNOD! Was loop 30
                         ILOC=INOD + (JNOD-1)*NMDNOD + (KNOD-1)*NMDNOD*NMDNOD
                         !
                         M(GPOI,ILOC) = &
                              SPECFU(NDIFF,LXGP,INOD,NMDNOD,IPOLY,NODPOS)  &
                              *SPECFU(NDIFF,LYGP,JNOD,NMDNOD,IPOLY,NODPOS) &
                              *SPECFU(NDIFF,LZGP,KNOD,NMDNOD,IPOLY,NODPOS)
                         !
                      end do ! Was loop 30
                   end do ! Was loop 30
                end do ! Was loop 30
             end do ! Was loop 102
          end do ! Was loop 102
       end do ! Was loop 102
    ENDIF
    !
    IF(D2) THEN
!       return
       print *,'here 1.1'
       NDGI  =INT((NGI**(1./2.))+0.1)
       NDNOD =INT((NLOC**(1./2.))+0.1)
       NMDNOD=INT((MLOC**(1./2.))+0.1)
       print *,'ndgi,ndnod,nmdnod:',ndgi,ndnod,nmdnod
       !
       ! Find the roots of the quadrature points and nodes
       ! also get the weights.
       print *,'here 1.2'
       CALL GTROOT(IPOLY,IQADRA,WEIT,NODPOS,QUAPOS,NDGI,NDNOD)
       print *,'here 1.3'
!       return
       do  IGQ=1,NDGI! Was loop 10
          do  IGP=1,NDGI! Was loop 10
             GPOI=IGP + (IGQ-1)*NDGI
             !
             WEIGHT(GPOI)=WEIT(IGP)*WEIT(IGQ)
             !
             LXGP=QUAPOS(IGP)
             LYGP=QUAPOS(IGQ)
             ! NB If TRUE in function RGPTWE then return the Gauss-pt weight
             ! else return the Gauss-pt.
             !
             do  JNOD=1,NDNOD! Was loop 120
                do  INOD=1,NDNOD! Was loop 120
                   ILOC=INOD + (JNOD-1)*NDNOD
                   !
                   N(GPOI,ILOC) = &
                        SPECFU(NDIFF,LXGP,INOD,NDNOD,IPOLY,NODPOS) &
                        *SPECFU(NDIFF,LYGP,JNOD,NDNOD,IPOLY,NODPOS)
                   !
                   NLX(GPOI,ILOC) = &
                        SPECFU(DIFF, LXGP,INOD,NDNOD,IPOLY,NODPOS) &
                        *SPECFU(NDIFF,LYGP,JNOD,NDNOD,IPOLY,NODPOS)
                   !
                   NLY(GPOI,ILOC) = &
                        SPECFU(NDIFF,LXGP,INOD,NDNOD,IPOLY,NODPOS) &
                        *SPECFU(DIFF, LYGP,JNOD,NDNOD,IPOLY,NODPOS)
                   !
                end do ! Was loop 120
             end do ! Was loop 120
          end do ! Was loop 10
       end do ! Was loop 10
       print *,'here 1.4'
!       return
       !
       ! Find the roots of the quadrature points and nodes
       ! also get the weights.
       CALL GTROOT(IPOLY,IQADRA,WEIT,NODPOS,QUAPOS,NDGI,NMDNOD)
       print *,'here 1.5'
       do  IGQ=1,NDGI! Was loop 11
          do  IGP=1,NDGI! Was loop 11
             GPOI=IGP + (IGQ-1)*NDGI
             LXGP=QUAPOS(IGP)
             LYGP=QUAPOS(IGQ)
             do  JNOD=1,NMDNOD! Was loop 130
                do  INOD=1,NMDNOD! Was loop 130
                   ILOC=INOD + (JNOD-1)*NMDNOD
                   !
                   M(GPOI,ILOC) = &
                        SPECFU(NDIFF,LXGP,INOD,NMDNOD,IPOLY,NODPOS) &
                        *SPECFU(NDIFF,LYGP,JNOD,NMDNOD,IPOLY,NODPOS)
                   !
                end do ! Was loop 130
             end do ! Was loop 130
             !
          end do ! Was loop 11
       end do ! Was loop 11
       print *,'here 1.6 d2=',d2
    ENDIF
    !
    !
    IF((.NOT.D2).AND.(.NOT.D3).and.(.not.D4)) THEN
       NDGI  =NGI
       NDNOD =NLOC
       NMDNOD=MLOC
       !
       ! Find the roots of the quadrature points and nodes
       ! also get the weights.
       CALL GTROOT(IPOLY,IQADRA,WEIT,NODPOS,QUAPOS,NDGI,NDNOD)
       !ewrite(3,*)'NDGI,NDNOD,NLOC:',NDGI,NDNOD,NLOC
       !ewrite(3,*)'WEIT(1:ndgi):',WEIT(1:ndgi)
       !ewrite(3,*)'NODPOS(1:ndnod):',NODPOS(1:ndnod)
       !ewrite(3,*)'QUAPOS(1:ndgi):',QUAPOS(1:ndgi)
       do  IGP=1,NDGI! Was loop 1000
          GPOI=IGP
          !
          WEIGHT(GPOI)=WEIT(IGP)
          !
          LXGP=QUAPOS(IGP)
          ! NB If TRUE in function RGPTWE then return the Gauss-pt weight
          ! else return the Gauss-pt.
          !
          do  INOD=1,NDNOD! Was loop 12000
             ILOC=INOD
             !
             N(GPOI,ILOC) = &
                  SPECFU(NDIFF,LXGP,INOD,NDNOD,IPOLY,NODPOS)
             !
             NLX(GPOI,ILOC) = &
                  SPECFU(DIFF, LXGP,INOD,NDNOD,IPOLY,NODPOS)
             !ewrite(3,*)'ILOC,GPOI,N(GPOI,ILOC),NLX(GPOI,ILOC):', &
             !         ILOC,GPOI,N(GPOI,ILOC),NLX(GPOI,ILOC)
             !
          end do ! Was loop 12000
       end do ! Was loop 1000
       !ewrite(3,*)'n WEIGHT:',WEIGHT
       !
       ! Find the roots of the quadrature points and nodes
       ! also get the weights.
       !ewrite(3,*)'this is for m which we dont care about:'
       CALL GTROOT(IPOLY,IQADRA,WEIT,NODPOS,QUAPOS,NDGI,NMDNOD)
       do  IGP=1,NDGI! Was loop 1100
          GPOI=IGP
          LXGP=QUAPOS(IGP)
          do  INOD=1,NMDNOD! Was loop 13000
             ILOC=INOD
             !
             M(GPOI,ILOC) = &
                  SPECFU(NDIFF,LXGP,INOD,NMDNOD,IPOLY,NODPOS)
             !
          end do ! Was loop 13000
          !
       end do ! Was loop 1100
       !ewrite(3,*)'...finished this is for m which we dont care about:'
    ENDIF
    print *,'just leaving spectr'
  END SUBROUTINE SPECTR




  SUBROUTINE GTROOT(IPOLY,IQADRA,WEIT,NODPOS,QUAPOS,NDGI,NDNOD)
    IMPLICIT NONE
    INTEGER , intent(in) :: IPOLY,IQADRA,NDGI,NDNOD
    REAL , dimension(NDNOD), intent(inout) :: NODPOS
    REAL , dimension(NDGI), intent(inout) :: WEIT,QUAPOS
    !Local variables
    LOGICAL :: GETNDP
    !     This sub returns the weights WEIT the quadrature points QUAPOS and
    !     the node points NODPOS.
    !     NODAL POISTIONS ******
    !     NB if GETNDP then find the nodal positions
!    print *,'IQADRA=',IQADRA
!    stop 38
    GETNDP=.TRUE.
    !     Compute standard Lagrange nodal points
    IF(IQADRA.EQ.1) CALL LAGROT(NODPOS,NODPOS,NDNOD,GETNDP)
    !     Compute Chebyshev-Gauss-Lobatto nodal points.
    IF(IQADRA.EQ.2) CALL CHEROT(NODPOS,NODPOS,NDNOD,GETNDP)
    IF(IQADRA.EQ.3) CALL CHEROT(NODPOS,NODPOS,NDNOD,GETNDP)
    !     Compute Legendre-Gauss-Lobatto nodal points.
    IF(IQADRA.EQ.4) CALL LEGROT(NODPOS,NODPOS,NDNOD,GETNDP)
    !
    !     QUADRATURE************
    GETNDP=.FALSE.
    !     Compute standard Gauss quadrature. weits and points
    IF(IQADRA.EQ.1) CALL LAGROT(WEIT,QUAPOS,NDGI,GETNDP)
    !     Compute Chebyshev-Gauss-Lobatto quadrature.
    IF(IQADRA.EQ.2) CALL CHEROT(WEIT,QUAPOS,NDGI,GETNDP)
    IF(IQADRA.EQ.3) CALL CHEROT(WEIT,QUAPOS,NDGI,GETNDP)
    !     Compute Legendre-Gauss-Lobatto quadrature.
    IF(IQADRA.EQ.4) CALL LEGROT(WEIT,QUAPOS,NDGI,GETNDP)
  END SUBROUTINE GTROOT




  REAL FUNCTION SPECFU(DIFF,LXGP,INOD,NDNOD,IPOLY,NODPOS)
    LOGICAL , intent(in):: DIFF
    INTEGER , intent(in) :: INOD,NDNOD,IPOLY
    REAL , intent(inout) :: LXGP
    real, dimension(NDNOD), intent(inout) :: NODPOS
    !     INOD contains the node at which the polynomial is associated with
    !     LXGP is the position at which the polynomial is to be avaluated.\
    !     If(DIFF) then find the D poly/DX.
! local variables (functions)
    real LAGRAN,CHEBY1,CHEBY2,LEGEND
    !
    IF(IPOLY.EQ.1) SPECFU=LAGRAN(DIFF,LXGP,INOD,NDNOD,NODPOS)
!         print *,'ipoly,NODPOS:',ipoly,NODPOS
!          stop 2921
    !
    IF(IPOLY.EQ.2) SPECFU=CHEBY1(DIFF,LXGP,INOD,NDNOD,NODPOS)
    !
    IF(IPOLY.EQ.3) SPECFU=CHEBY2(DIFF,LXGP,INOD,NDNOD,NODPOS)
    !
    IF(IPOLY.EQ.4) SPECFU=LEGEND(DIFF,LXGP,INOD,NDNOD,NODPOS)
  END FUNCTION SPECFU



  SUBROUTINE CHEROT(WEIT,QUAPOS,NDGI,GETNDP)
    IMPLICIT NONE
    INTEGER , intent(in) :: NDGI
    REAL, dimension(NDGI), intent(inout) :: WEIT, QUAPOS
    LOGICAL , intent(in) :: GETNDP
    !     This computes the weight and points for Chebyshev-Gauss-Lobatto quadrature.
    !     See page 67 of:Spectral Methods in Fluid Dynamics, C.Canuto
    !     IF(GETNDP) then get the POSITION OF THE NODES
    !     AND DONT BOTHER WITH THE WEITS.
    !Local variables
    real , PARAMETER :: PIE=3.141569254
    INTEGER :: IG,J
    !
    IF(.NOT.GETNDP) THEN
       !     THE WEIGHTS...
       WEIT(1)       =PIE/(2.*(NDGI-1))
       WEIT(NDGI)=PIE/(2.*(NDGI-1))
       do IG=2,NDGI-1
          WEIT(IG)   =PIE/(NDGI-1)
       END DO
    ENDIF
    !
    !     The quad points...
    do IG=1,NDGI
       J=IG-1
       QUAPOS(IG)=COS(PIE*REAL(J)/REAL(NDGI-1))
    END DO
  END SUBROUTINE CHEROT




  SUBROUTINE LEGROT(WEIT,QUAPOS,NDGI,GETNDP)
    IMPLICIT NONE
    !     This computes the weight and points for Chebyshev-Gauss-Lobatto quadrature.
    !     See page 69 of:Spectral Methods in Fluid Dynamics, C.Canuto
    !     IF(GETNDP) then get the POSITION OF THE NODES
    !     AND DONT BOTHER WITH THE WEITS.
    INTEGER , intent(in) :: NDGI
    REAL , dimension(NDGI) , intent(inout) :: WEIT,QUAPOS
    LOGICAL , intent(in) :: GETNDP
    !Local variables
    INTEGER :: N,IG
    real :: plegen
    !     Work out the root's i.e the quad positions first.
    CALL LROOTS(QUAPOS,NDGI)
    IF(.NOT.GETNDP) THEN
       !     THE WEIGHTS...
       N=NDGI-1
       do IG=1,NDGI
          WEIT(IG)=2./(REAL(N*(N+1))*PLEGEN(QUAPOS(IG),NDGI-1)**2)
       END DO
    ENDIF
  END SUBROUTINE LEGROT



  real function PLEGEN(LX,K)
    REAL , intent(in):: LX
    INTEGER , intent(in) :: K
    !Local variables
    REAL :: R
    INTEGER :: L
    R=0.
    DO L=0,INT(K/2)
       R=R+((-1)**L)*binomial_coefficient(K,L) &
                      *binomial_coefficient(2*K-2*L,K)*LX**(K-2*L)
    END DO
    PLEGEN=R/REAL(2**K)
  end function plegen



  real function binomial_coefficient(K,L)
    !<! Calculate binomial coefficients
    integer factorial

    INTEGER K,L
    binomial_coefficient=factorial(K)/(factorial(L)*factorial(K-L))

  end function binomial_coefficient



  SUBROUTINE LROOTS(QUAPOS,NDGI)
    IMPLICIT NONE
    INTEGER, intent(in) :: NDGI
    REAL , dimension(NDGI), intent(inout) :: QUAPOS
    !Local variables
    REAL ALPHA,BETA,RKEEP
    INTEGER N,I
    !     This sub works out the Gauss-Lobatto-Legendre roots.
    ALPHA = 0.
    BETA   = 0.
    !
    N=NDGI-1
    CALL JACOBL(N,ALPHA,BETA,QUAPOS)
    !     Now reverse ordering.
    do I=1,INT(0.1+ NDGI/2)
       RKEEP=QUAPOS(I)
       QUAPOS(I)=QUAPOS(NDGI+1-I)
       QUAPOS(NDGI+1-I)=RKEEP
    END DO
  END SUBROUTINE LROOTS




  REAL FUNCTION CHEBY1(DIFF,LX,INOD,NDNOD,NODPOS)
    IMPLICIT NONE
    INTEGER, intent(in) :: NDNOD,INOD
    REAL, intent(in) :: LX
    real, dimension(NDNOD), intent(in) :: NODPOS
    LOGICAL, intent(in)  ::  DIFF
    !Local variables
    LOGICAL  ::  DIFF2
    real :: RNX
    real tcheb
    !     If DIFF then returns the spectral function DIFFERENTIATED W.R.T X
    !     associated.
    !     This function returns the spectral function associated
    !     with node INOD at POINT LX
    !     NDNOD=no of nodes in 1-D.
    !     NDGI=no of Gauss pts in 1-D.
    !     NB The nodes are at the points COS(pie*J/2.) j=0,..,ndgi-1
    REAL CJ,CN,HX,XPT
    INTEGER NX,N

    NX=NDNOD-1
    RNX=REAL(NX)
    DIFF2=.FALSE.
    !
    CJ=1.
    IF((INOD.EQ.1).OR.(INOD.EQ.NDNOD)) CJ=2.
    HX=0.
    !
    CN=2.
    N=0
    XPT=NODPOS(INOD)
    HX=HX + ( (2./RNX)/(CJ*CN) )*TCHEB(N,XPT,.FALSE.,DIFF2)&
         &     *TCHEB(N,LX,DIFF,DIFF2)
    CN=1.
    do  N=1,NX-1! Was loop 10
       HX=HX + ( (2./RNX)/(CJ*CN) )*TCHEB(N,XPT,.FALSE.,DIFF2)&
            &        *TCHEB(N,LX,DIFF,DIFF2)
    end do ! Was loop 10
    CN=2.
    N=NX
    HX=HX + ( (2./RNX)/(CJ*CN))*TCHEB(N,XPT,.FALSE.,DIFF2)&
         &     *TCHEB(N,LX,DIFF,DIFF2)
    CHEBY1=HX
  END FUNCTION CHEBY1




  REAL FUNCTION CHEBY2(DIFF,LX,INOD,NDNOD,NODPOS)
    IMPLICIT NONE
    INTEGER , intent(in) :: NDNOD,INOD
    REAL, dimension(NDNOD), intent(in) :: NODPOS
    LOGICAL , intent(in) :: DIFF
    REAL, intent(in) :: LX
    !Local variables
    real :: R,RR,RCONST
    real :: tcheb
    !     If DIFF then returns the spectral function DIFFERENTIATED W.R.T X
    !     associated.
    !     This function returns the spectral function associated
    !     with node INOD at POINT LX
    !     NDNOD=no of nodes in 1-D.
    !     NDGI=no of Gauss pts in 1-D.
    !     NB The nodes are at the points COS(pie*J/2.) j=0,..,ndgi-1
    REAL CJ,CN,HX,XPT
    IF(.NOT.DIFF) THEN
       IF(ABS(LX-NODPOS(INOD)).LT.1.E-10) THEN
          CHEBY2=1.
       ELSE
          R=(-1.)**(INOD)*(1.-LX**2)&
               &       *TCHEB(NDNOD-1,LX,.TRUE.,.FALSE.)
          CJ=1.
          IF((INOD.EQ.1).OR.(INOD.EQ.NDNOD)) CJ=2.
          R=R/(CJ*(NDNOD-1)**2 *(LX-NODPOS(INOD)) )
          CHEBY2=R
       ENDIF
    ELSE
       IF(ABS(LX-NODPOS(INOD)).LT.1.E-10) THEN
          IF(ABS(LX+1.).LT.1.E-10) THEN
             CHEBY2= (2.*(NDNOD-1)**2+1)/6.
          ELSE
             IF(ABS(LX-1.).LT.1.E-10) THEN
                CHEBY2=-(2.*(NDNOD-1)**2+1)/6.
             ELSE
                CHEBY2=-LX/(2.*(1.-LX**2))
             ENDIF
          ENDIF
       ELSE
          R=(-1.)**(INOD)*(1.-LX**2)&
               &          *TCHEB(NDNOD-1,LX,.FALSE.,.TRUE.)
          CJ=1.
          IF((INOD.EQ.1).OR.(INOD.EQ.NDNOD)) CJ=2.
          R=R/(CJ*(NDNOD-1)**2 *(LX-NODPOS(INOD)) )
          RR=(-1.)**(INOD)*TCHEB(NDNOD-1,LX,.TRUE.,.FALSE.)&
               &           /(CJ*REAL(NDNOD-1)**2)
          RCONST=-(2*LX+(1.-LX**2)/(LX-NODPOS(INOD) ) )&
               &              /(LX-NODPOS(INOD))
          RR=RR*RCONST
          CHEBY2=RR
       ENDIF
    ENDIF
  END FUNCTION CHEBY2



  REAL FUNCTION TCHEB(N,XPT,DIFF,DIFF2)
    !      use math_utilities
    IMPLICIT NONE
    LOGICAL , intent(in) :: DIFF,DIFF2
    integer, intent(in) :: N
    real, intent(in) :: XPT
    !Local variables
    REAL :: DDT,DT,T,TTEMP,TM1,DTM1,DTEMP,R,RR
    INTEGER :: K,L
    INTEGER :: NI
    integer :: factorial
    ! If DIFF then return the n'th Chebyshef polynomial
    ! differentiated w.r.t x.
    ! If DIFF2 then form the 2'nd derivative.
    ! This sub returns the value of the K'th Chebyshef polynomial at a
    ! point XPT.
    !
    ! This formula can be found in:Spectral Methods for Fluid Dynamics, page 66
    K=N
    !
    IF((.NOT.DIFF).AND.(.NOT.DIFF2)) THEN
       IF(N.EQ.0) T=1.
       IF(N.EQ.1) T=XPT
       IF(N.GT.1) THEN
          TM1=1.
          T=XPT
          do  NI=2,N! Was loop 110
             TTEMP=T
             T=2.*XPT*TTEMP - TM1
             TM1=TTEMP
          end do ! Was loop 110
       ENDIF
       TCHEB=T
    ENDIF
    !
    IF(DIFF.AND.(.NOT.DIFF2)) THEN
       ! This part forms the differential w.r.t x.
       IF(N.EQ.0) DT=0.
       IF(N.EQ.1) THEN
          T=XPT
          DT=1.
       ENDIF
       IF(N.EQ.2) THEN
          T=2.*XPT*XPT -1
          DT=4.*XPT
       ENDIF
       IF(N.GT.2) THEN
          TM1=XPT
          DTM1=1.
          T=2.*XPT*XPT -1
          DT=4.*XPT
          do  NI=2,N! Was loop 10
             TTEMP=T
             T=2.*XPT*TTEMP - TM1
             TM1=TTEMP
             !
             DTEMP=DT
             DT=2.*XPT*DTEMP+2.*TTEMP-DTM1
             DTM1=DTEMP
          end do ! Was loop 10
       ENDIF
       TCHEB=DT
    ENDIF
    !
    IF(DIFF2) THEN
       IF(N.LE.1) THEN
          DDT=0.
       ELSE
          R=0.
          do  L=0,INT(0.1+K/2)! Was loop 50
             RR=(-1.)**K*factorial(K-L-1)/(factorial(L)*factorial(K-2*L))
             RR=RR*2**(K-2*L)
             ! The following is for 2'nd derivative.
             RR=RR*(K-2.*L)*(K-2.*L-1)*XPT**(K-2*L-2)
             R=R+RR
          end do ! Was loop 50
          DDT=R*REAL(K)*0.5
       ENDIF
       TCHEB=DDT
    ENDIF
  END FUNCTION TCHEB



  recursive function factorial(n) result(f)
    ! Calculate n!
    integer :: f
    integer, intent(in) :: n

    if (n==0) then
       f=1
    else
       f=n*factorial(n-1)
    end if

  end function factorial



  REAL FUNCTION LEGEND(DIFF,LX,INOD,NDNOD,NODPOS)
    INTEGER , intent(in) :: INOD,NDNOD
    REAL, dimension(NDNOD), intent(in) :: NODPOS
    REAL, intent(in) :: LX
    LOGICAL , intent(in) :: DIFF
    !     If DIFF then returns the spectral function DIFFERENTIATED W.R.T X
    !     associated.
    !     This function returns the spectral function associated
    !     with node INOD at POINT LX
    !     NDNOD=no of nodes in 1-D.
    !     NDGI=no of Gauss pts in 1-D.
    !     NB The nodes are at the points COS(pie*J/2.) j=0,..,ndgi-1
    REAL CJ,CN,HX,XPT
    LEGEND=1.
  END FUNCTION LEGEND



! ************************************************************************
! *********TRIANGLES AND TETS ********************************************
! ************************************************************************




  SUBROUTINE SHATRInew(L1, L2, L3, L4, WEIGHT, &
       NLOC,NGI,ndim,  N,NLX_ALL)
    ! Interface to SHATRIold using the new style variables
    IMPLICIT NONE
    INTEGER , intent(in) :: NLOC,NGI,ndim
    REAL , dimension(ngi), intent(in) :: L1, L2, L3, L4
    REAL , dimension(ngi), intent(inout) :: WEIGHT
    REAL , dimension(ngi, nloc ), intent(inout) ::N
    real, dimension (ngi,ndim,nloc), intent(inout) :: NLX_ALL
! local variables...
    REAL Nold(nloc, ngi ),NLXold(nloc, ngi ),NLYold(nloc, ngi ),NLZold(nloc, ngi )
    integer iloc

!      print *,'**going into SHATRIold...'
!      print *,'just before ndim,nloc,ngi:',ndim,nloc,ngi
!    call SHATRIold(L1, L2, L3, L4, WEIGHT, size(NLX_ALL,1)==3, &
    call SHATRIold(L1, L2, L3, L4, WEIGHT, ndim==3, &
         NLOC,NGI,  &
         Nold,NLXold,NLYold,NLZold)
!         N,NLX_ALL(:,1,:),NLX_ALL(:,2,:),NLX_ALL(:,ndim,:))
    do iloc=1,nloc
       N(:,iloc) = Nold(iloc,:)
       NLX_ALL(:,1,iloc)=NLXold(iloc,:)
       NLX_ALL(:,2,iloc)=NLYold(iloc,:)
       if(ndim==3) NLX_ALL(:,3,iloc) = NLZold(iloc,:)
    end do
!    print *,'n:',n
!    print *,'nlx_all:',nlx_all
!    print *,'nloc,ngi:',nloc,ngi
!    print *,'leaving SHATRInew'

  end subroutine SHATRInew
  !
  !
  SUBROUTINE SHATRIold(L1, L2, L3, L4, WEIGHT, D3, &
       NLOC,NGI,  &
       N,NLX,NLY,NLZ)
    ! Work out the shape functions and there derivatives...
    IMPLICIT NONE
    INTEGER , intent(in) :: NLOC,NGI
    LOGICAL , intent(in) :: D3
    REAL , dimension(ngi), intent(in) :: L1, L2, L3, L4
    REAL , dimension(ngi), intent(inout) :: WEIGHT
    REAL , dimension(nloc, ngi ), intent(inout) ::N,NLX,NLY,NLZ
    ! Local variables...
    INTEGER ::  GI
    !
    IF(.NOT.D3) THEN
       ! Assume a triangle...
       !
       IF(NLOC.EQ.1) THEN
          Loop_Gi_Nloc1: DO GI=1,NGI
             N(1,GI)=1.0
             NLX(1,GI)=0.0
             NLY(1,GI)=0.0
          end DO Loop_Gi_Nloc1
       ELSE IF((NLOC.EQ.3).OR.(NLOC.EQ.4)) THEN
          Loop_Gi_Nloc3_4: DO GI=1,NGI
             N(1,GI)=L1(GI)
             N(2,GI)=L2(GI)
             N(3,GI)=L3(GI)
             !
             NLX(1,GI)=1.0
             NLX(2,GI)=0.0
             NLX(3,GI)=-1.0
             !
             NLY(1,GI)=0.0
             NLY(2,GI)=1.0
             NLY(3,GI)=-1.0
             IF(NLOC.EQ.4) THEN
                ! Bubble function...
                !alpha == 1 behaves better than the correct value of 27. See Osman et al. 2019
                N(4,GI)  =1. * L1(GI)*L2(GI)*L3(GI)
                NLX(4,GI)=1. * L2(GI)*(1.-L2(GI))-2.*L1(GI)*L2(GI)
                NLY(4,GI)=1. * L1(GI)*(1.-L1(GI))-2.*L1(GI)*L2(GI)
             ENDIF
          end DO Loop_Gi_Nloc3_4
       ELSE IF((NLOC.EQ.6).OR.(NLOC.EQ.7)) THEN
          Loop_Gi_Nloc_6_7: DO GI=1,NGI
             N(1,GI)=(2.*L1(GI)-1.)*L1(GI)
             N(2,GI)=(2.*L2(GI)-1.)*L2(GI)
             N(3,GI)=(2.*L3(GI)-1.)*L3(GI)
             !
             N(4,GI)=4.*L1(GI)*L2(GI)
             N(5,GI)=4.*L2(GI)*L3(GI)
             N(6,GI)=4.*L1(GI)*L3(GI)

             !
             ! nb L1+L2+L3+L4=1
             ! x-derivative...
             NLX(1,GI)=4.*L1(GI)-1.
             NLX(2,GI)=0.
             NLX(3,GI)=-4.*(1.-L2(GI))+4.*L1(GI) + 1.
             !
             NLX(4,GI)=4.*L2(GI)
             NLX(5,GI)=-4.*L2(GI)
             NLX(6,GI)=4.*(1.-L2(GI))-8.*L1(GI)
             !
             ! y-derivative...
             NLY(1,GI)=0.
             NLY(2,GI)=4.*L2(GI)-1.0
             NLY(3,GI)=-4.*(1.-L1(GI))+4.*L2(GI) + 1.
             !
             NLY(4,GI)=4.*L1(GI)
             NLY(5,GI)=4.*(1.-L1(GI))-8.*L2(GI)
             NLY(6,GI)=-4.*L1(GI)
             IF(NLOC.EQ.7) THEN
                ! Bubble function...
                N(7,GI)  =L1(GI)*L2(GI)*L3(GI)
                NLX(7,GI)=L2(GI)*(1.-L2(GI))-2.*L1(GI)*L2(GI)
                NLY(7,GI)=L1(GI)*(1.-L1(GI))-2.*L1(GI)*L2(GI)
             ENDIF
          END DO Loop_Gi_Nloc_6_7
          ! ENDOF IF(NLOC.EQ.6) THEN...
       ELSE IF(NLOC==10) THEN ! Cubic triangle...
          ! get the shape functions for a cubic triangle...
          call shape_triangle_cubic( l1, l2, l3, l4, weight, d3, &
               nloc, ngi, &
               n, nlx, nly, nlz )

       ELSE ! has not found the element shape functions
          stop 811
       ENDIF
       !
       ! ENDOF IF(.NOT.D3) THEN
    ENDIF
    !
    !
    IF(D3) THEN
       ! Assume a tet...
       ! This is for 5 point quadrature.
       IF((NLOC.EQ.10).OR.(NLOC.EQ.11)) THEN
          Loop_Gi_Nloc_10_11: DO GI=1,NGI
             !ewrite(3,*)'gi,L1(GI),L2(GI),L3(GI),L4(GI):',gi,L1(GI),L2(GI),L3(GI),L4(GI)
             N(1,GI)=(2.*L1(GI)-1.)*L1(GI)
             N(3,GI)=(2.*L2(GI)-1.)*L2(GI)
             N(5,GI)=(2.*L3(GI)-1.)*L3(GI)
             N(10,GI)=(2.*L4(GI)-1.)*L4(GI)

             !if(L1(GI).gt.-1.93) ewrite(3,*)'gi,L1(GI), L2(GI), L3(GI), L4(GI),N(1,GI):', &
             !                            gi,L1(GI), L2(GI), L3(GI), L4(GI),N(1,GI)
             !
             !
             N(2,GI)=4.*L1(GI)*L2(GI)
             N(6,GI)=4.*L1(GI)*L3(GI)
             N(7,GI)=4.*L1(GI)*L4(GI)
             !
             N(4,GI) =4.*L2(GI)*L3(GI)
             N(9,GI) =4.*L3(GI)*L4(GI)
             N(8,GI)=4.*L2(GI)*L4(GI)
             ! nb L1+L2+L3+L4=1
             ! x-derivative...
             NLX(1,GI)=4.*L1(GI)-1.
             NLX(3,GI)=0.
             NLX(5,GI)=0.
             NLX(10,GI)=-4.*(1.-L2(GI)-L3(GI))+4.*L1(GI) + 1.
             !if(L1(GI).gt.-1.93) ewrite(3,*)'Nlx(1,GI):', &
             !     Nlx(1,GI)
             !
             NLX(2,GI)=4.*L2(GI)
             NLX(6,GI)=4.*L3(GI)
             NLX(7,GI)=4.*(L4(GI)-L1(GI))
             !
             NLX(4,GI) =0.
             NLX(9,GI) =-4.*L3(GI)
             NLX(8,GI)=-4.*L2(GI)
             !
             ! y-derivative...
             NLY(1,GI)=0.
             NLY(3,GI)=4.*L2(GI)-1.0
             NLY(5,GI)=0.
             NLY(10,GI)=-4.*(1.-L1(GI)-L3(GI))+4.*L2(GI) + 1.
             !
             NLY(2,GI)=4.*L1(GI)
             NLY(6,GI)=0.
             NLY(7,GI)=-4.*L1(GI)
             !
             NLY(4,GI) =4.*L3(GI)
             NLY(9,GI) =-4.*L3(GI)
             NLY(8,GI)=4.*(1-L1(GI)-L3(GI))-8.*L2(GI)
             !
             ! z-derivative...
             NLZ(1,GI)=0.
             NLZ(3,GI)=0.
             NLZ(5,GI)=4.*L3(GI)-1.
             NLZ(10,GI)=-4.*(1.-L1(GI)-L2(GI))+4.*L3(GI) + 1.
             !
             NLZ(2,GI)=0.
             NLZ(6,GI)=4.*L1(GI)
             NLZ(7,GI)=-4.*L1(GI)
             !
             NLZ(4,GI) =4.*L2(GI)
             NLZ(9,GI) =4.*(1.-L1(GI)-L2(GI))-8.*L3(GI)
             NLZ(8,GI)=-4.*L2(GI)
             IF(NLOC.EQ.11) THEN
                ! Bubble function...
                N(11,GI)  =L1(GI)*L2(GI)*L3(GI)*L4(GI)
                NLX(11,GI)=L2(GI)*L3(GI)*(1.-L2(GI)-L3(GI))-2.*L1(GI)*L2(GI)*L3(GI)
                NLY(11,GI)=L1(GI)*L3(GI)*(1.-L1(GI)-L3(GI))-2.*L1(GI)*L2(GI)*L3(GI)
                NLZ(11,GI)=L1(GI)*L2(GI)*(1.-L1(GI)-L2(GI))-2.*L1(GI)*L2(GI)*L3(GI)
             ENDIF
             !
          end DO Loop_Gi_Nloc_10_11
          ! ENDOF IF(NLOC.EQ.10) THEN...
       ENDIF
       !
       IF((NLOC.EQ.4).OR.(NLOC.EQ.5)) THEN
          Loop_Gi_Nloc_4_5: DO GI=1,NGI
             N(1,GI)=L1(GI)
             N(2,GI)=L2(GI)
             N(3,GI)=L3(GI)
             N(4,GI)=L4(GI)
             !
             NLX(1,GI)=1.0
             NLX(2,GI)=0
             NLX(3,GI)=0
             NLX(4,GI)=-1.0
             !
             NLY(1,GI)=0.0
             NLY(2,GI)=1.0
             NLY(3,GI)=0.0
             NLY(4,GI)=-1.0
             !
             NLZ(1,GI)=0.0
             NLZ(2,GI)=0.0
             NLZ(3,GI)=1.0
             NLZ(4,GI)=-1.0
             IF(NLOC.EQ.5) THEN
                ! Bubble function ...
                !alpha == 50 behaves better than the correct value of 256. See Osman et al. 2019
                N(5,GI)  = 50. * L1(GI)*L2(GI)*L3(GI)*L4(GI)
                NLX(5,GI)= 50. * L2(GI)*L3(GI)*(1.-L2(GI)-L3(GI))  &
                         -2.*L1(GI)*L2(GI)*L3(GI)
                NLY(5,GI)= 50. * L1(GI)*L3(GI)*(1.-L1(GI)-L3(GI))  &
                         -2.*L1(GI)*L2(GI)*L3(GI)
                NLZ(5,GI)= 50. * L1(GI)*L2(GI)*(1.-L1(GI)-L2(GI))  &
                         -2.*L1(GI)*L2(GI)*L3(GI)
             ENDIF
          end DO Loop_Gi_Nloc_4_5
       ENDIF
       !
       IF(NLOC.EQ.1) THEN
          Loop_Gi_Nloc_1: DO GI=1,NGI
             N(1,GI)=1.0
             NLX(1,GI)=0.0
             NLY(1,GI)=0.0
             NLZ(1,GI)=0.0
          end DO Loop_Gi_Nloc_1
       ENDIF
       !
       ! ENDOF IF(D3) THEN...
    ENDIF
    !
    RETURN
  END SUBROUTINE SHATRIold
  !
  !
  !
  !
  SUBROUTINE TRIQUAold(L1, L2, L3, L4, WEIGHT, D3,NGI)
    ! This sub calculates the local corrds L1, L2, L3, L4 and
    ! weights at the quadrature points.
    ! If D3 it does this for 3Dtetrahedra elements else
    ! triangular elements.
    IMPLICIT NONE
    INTEGER , intent(in):: NGI
    LOGICAL , intent(in) :: D3
    REAL , dimension(ngi) , intent(inout) ::L1, L2, L3, L4, WEIGHT
    ! Local variables...
    REAL :: ALPHA,BETA
    REAL :: ALPHA1,BETA1
    REAL :: ALPHA2,BETA2
    real :: rsum
    INTEGER I
    !
    IF(D3) THEN
       ! this is for a tetrahedra element...
       ! This is for one point.
       IF(NGI.EQ.1) THEN
          ! Degree of precision is 1
          DO I=1,NGI
             L1(I)=0.25
             L2(I)=0.25
             L3(I)=0.25
             L4(I)=0.25
             WEIGHT(I)=1.0
          END DO
       ENDIF

       IF(NGI.EQ.4) THEN
          ! Degree of precision is 2
          ALPHA=0.58541020
          BETA=0.13819660
          DO I=1,NGI
             L1(I)=BETA
             L2(I)=BETA
             L3(I)=BETA
             L4(I)=BETA
             WEIGHT(I)=0.25
          END DO
          L1(1)=ALPHA
          L2(2)=ALPHA
          L3(3)=ALPHA
          L4(4)=ALPHA
       ENDIF

       IF(NGI.EQ.5) THEN
          ! Degree of precision is 3
          L1(1)=0.25
          L2(1)=0.25
          L3(1)=0.25
          L4(1)=0.25
          WEIGHT(1)=-4./5.
          !
          DO I=2,NGI
             L1(I)=1./6.
             L2(I)=1./6.
             L3(I)=1./6.
             L4(I)=1./6.
             WEIGHT(I)=9./20.
          END DO
          L1(2)=0.5
          L2(3)=0.5
          L3(4)=0.5
          L4(5)=0.5
       ENDIF
       !
       IF(NGI.EQ.11) THEN
          ! Degree of precision is 4
          ALPHA=(1.+SQRT(5./14.))/4.0
          BETA =(1.-SQRT(5./14.))/4.0
          I=1
          L1(I)=0.25
          L2(I)=0.25
          L3(I)=0.25
          WEIGHT(I)=-6.*74.0/5625.0
          DO I=2,5
             L1(I)=1./14.
             L2(I)=1./14.
             L3(I)=1./14.
             WEIGHT(I)=6.*343./45000.
          END DO
          L1(2)=11./14.
          L2(3)=11./14.
          L3(4)=11./14.
          DO I=6,11
             L1(I)=ALPHA
             L2(I)=ALPHA
             L3(I)=ALPHA
             WEIGHT(I)=6.*56.0/2250.0
          END DO
          L3(6)=BETA
          L2(7)=BETA
          L2(8)=BETA
          L3(8)=BETA
          L1(9)=BETA
          L1(10)=BETA
          L3(10)=BETA
          L1(11)=BETA
          L2(11)=BETA
          ! ENDOF IF(NGI.EQ.11) THEN...
       ENDIF

        if (NGI == 15) then!Fith order quadrature
          ! Degree of precision is 5
         L1=(/0.2500000000000000, 0.0000000000000000, 0.3333333333333333, &
             0.3333333333333333, 0.3333333333333333, &
             0.7272727272727273, 0.0909090909090909, 0.0909090909090909, &
             0.0909090909090909, 0.4334498464263357, &
             0.0665501535736643, 0.0665501535736643, 0.0665501535736643, &
             0.4334498464263357, 0.4334498464263357/)
         L2=(/0.2500000000000000, 0.3333333333333333, 0.3333333333333333, &
             0.3333333333333333, 0.0000000000000000, &
             0.0909090909090909, 0.0909090909090909, 0.0909090909090909, &
             0.7272727272727273, 0.0665501535736643, &
             0.4334498464263357, 0.0665501535736643, 0.4334498464263357, &
             0.0665501535736643, 0.4334498464263357/)
         L3=(/0.2500000000000000, 0.3333333333333333, 0.3333333333333333, &
             0.0000000000000000, 0.3333333333333333, &
             0.0909090909090909, 0.0909090909090909, 0.7272727272727273, &
             0.0909090909090909, 0.0665501535736643, &
             0.0665501535736643, 0.4334498464263357, 0.4334498464263357, &
             0.4334498464263357, 0.0665501535736643/)
         !We divide the weights later by 6
         WEIGHT=(/0.1817020685825351, 0.0361607142857143, 0.0361607142857143, &
             0.0361607142857143, 0.0361607142857143, &
             0.0698714945161738, 0.0698714945161738, 0.0698714945161738, &
             0.0698714945161738, 0.0656948493683187, &
             0.0656948493683187, 0.0656948493683187, 0.0656948493683187, &
             0.0656948493683187, 0.0656948493683187/)
       end if


       if (NGI == 45) then!Eighth order quadrature, for bubble shape functions or P3
         !Obtained from: https://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tet/quadrature_rules_tet.html
         !Here to get triangle quadrature sets:
         !https://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tri/quadrature_rules_tri.html
         ! Degree of precision is 8
        L1=(/0.2500000000000000,0.6175871903000830,0.1274709365666390,0.1274709365666390,&
        0.1274709365666390,0.9037635088221031,&
        0.0320788303926323,0.0320788303926323,0.0320788303926323,0.4502229043567190,&
        0.0497770956432810,0.0497770956432810,&
        0.0497770956432810,0.4502229043567190,0.4502229043567190,0.3162695526014501,&
        0.1837304473985499,0.1837304473985499,&
        0.1837304473985499,0.3162695526014501,0.3162695526014501,0.0229177878448171,&
        0.2319010893971509,0.2319010893971509,&
        0.5132800333608811,0.2319010893971509,0.2319010893971509,0.2319010893971509,&
        0.0229177878448171,0.5132800333608811,&
        0.2319010893971509,0.0229177878448171,0.5132800333608811,0.7303134278075384,&
        0.0379700484718286,0.0379700484718286,&
        0.1937464752488044,0.0379700484718286,0.0379700484718286,0.0379700484718286,&
        0.7303134278075384,0.1937464752488044,&
        0.0379700484718286,0.7303134278075384,0.1937464752488044/)
        L2=(/0.2500000000000000,0.1274709365666390,0.1274709365666390,0.1274709365666390,&
        0.6175871903000830,0.0320788303926323,&
        0.0320788303926323,0.0320788303926323,0.9037635088221031,0.0497770956432810,&
        0.4502229043567190,0.0497770956432810,&
        0.4502229043567190,0.0497770956432810,0.4502229043567190,0.1837304473985499,&
        0.3162695526014501,0.1837304473985499,&
        0.3162695526014501,0.1837304473985499,0.3162695526014501,0.2319010893971509,&
        0.0229177878448171,0.2319010893971509,&
        0.2319010893971509,0.5132800333608811,0.2319010893971509,0.0229177878448171,&
        0.5132800333608811,0.2319010893971509,&
        0.5132800333608811,0.2319010893971509,0.0229177878448171,0.0379700484718286,&
        0.7303134278075384,0.0379700484718286,&
        0.0379700484718286,0.1937464752488044,0.0379700484718286,0.7303134278075384,&
        0.1937464752488044,0.0379700484718286,&
        0.1937464752488044,0.0379700484718286,0.7303134278075384/)
        L3=(/0.2500000000000000,0.1274709365666390,0.1274709365666390,0.6175871903000830,&
        0.1274709365666390,0.0320788303926323,&
        0.0320788303926323,0.9037635088221031,0.0320788303926323,0.0497770956432810,&
        0.0497770956432810,0.4502229043567190,&
        0.4502229043567190,0.4502229043567190,0.0497770956432810,0.1837304473985499,&
        0.1837304473985499,0.3162695526014501,&
        0.3162695526014501,0.3162695526014501,0.1837304473985499,0.2319010893971509,&
        0.2319010893971509,0.0229177878448171,&
        0.2319010893971509,0.2319010893971509,0.5132800333608811,0.5132800333608811,&
        0.2319010893971509,0.0229177878448171,&
        0.0229177878448171,0.5132800333608811,0.2319010893971509,0.0379700484718286,&
        0.0379700484718286,0.7303134278075384,&
        0.0379700484718286,0.0379700484718286,0.1937464752488044,0.1937464752488044,&
        0.0379700484718286,0.7303134278075384,&
        0.7303134278075384,0.1937464752488044,0.0379700484718286/)
        !We divide the weights later by 6
        WEIGHT=(/-0.2359620398477557,0.0244878963560562,0.0244878963560562,0.0244878963560562,&
        0.0244878963560562,0.0039485206398261,&
        0.0039485206398261,0.0039485206398261,0.0039485206398261,0.0263055529507371,&
        0.0263055529507371,0.0263055529507371,&
        0.0263055529507371,0.0263055529507371,0.0263055529507371,0.0829803830550589,&
        0.0829803830550589,0.0829803830550589,&
        0.0829803830550589,0.0829803830550589,0.0829803830550589,0.0254426245481023,&
        0.0254426245481023,0.0254426245481023,&
        0.0254426245481023,0.0254426245481023,0.0254426245481023,0.0254426245481023,&
        0.0254426245481023,0.0254426245481023,&
        0.0254426245481023,0.0254426245481023,0.0254426245481023,0.0134324384376852,&
        0.0134324384376852,0.0134324384376852,&
        0.0134324384376852,0.0134324384376852,0.0134324384376852,0.0134324384376852,&
        0.0134324384376852,0.0134324384376852,&
        0.0134324384376852,0.0134324384376852,0.0134324384376852/)
      end if

       DO I=1,NGI
          L4(I)=1.0-L1(I)-L2(I)-L3(I)
       END DO

       ! Now multiply by 1/6. to get weigts correct...
       DO I=1,NGI
          WEIGHT(I)=WEIGHT(I)/6.
       END DO
       ! ENDOF IF(D3) THEN...
    ENDIF
    !
    IF(.NOT.D3) THEN
       ! 2-D TRAINGULAR ELEMENTS...
       IF(NGI.EQ.1) THEN
          ! LINEAR
          I=1
          L1(I)=1./3.
          L2(I)=1./3.
          WEIGHT(I)=1.0
       ENDIF
       !
       IF(NGI.EQ.3) THEN
          ! QUADRASTIC
          DO I=1,NGI
             L1(I)=0.5
             L2(I)=0.5
             WEIGHT(I)=1.0/3.0
          END DO
          L1(2)=0.0
          L2(3)=0.0
       ENDIF
       !
       IF(NGI.EQ.4) THEN
          ! CUBIC
          I=1
          L1(I)=1./3.
          L2(I)=1./3.
          WEIGHT(I)=-27./48.
          DO I=2,NGI
             L1(I)=0.2
             L2(I)=0.2
             WEIGHT(I)=25./48.
          END DO
          L1(1)=0.6
          L2(2)=0.6
       ENDIF
       !
       IF(NGI.EQ.7) THEN
          ! QUNTIC
          ALPHA1=0.0597158717
          BETA1 =0.4701420641
          ALPHA2=0.7974269853
          BETA2 =0.1012865073
          I=1
          L1(I)=1./3.
          L2(I)=1./3.
          WEIGHT(I)=0.225
          DO I=2,4
             L1(I)=BETA1
             L2(I)=BETA1
             WEIGHT(I)=0.1323941527
          END DO
          L1(2)=ALPHA1
          L2(4)=ALPHA1
          DO I=5,7
             L1(I)=BETA2
             L2(I)=BETA2
             WEIGHT(I)=0.1259391805
          END DO
          L1(5)=ALPHA2
          L2(6)=ALPHA2
          ! ENDOF IF(NGI.EQ.7) THEN...
       ENDIF

       IF(NGI.EQ.14) THEN
          ! 5th order quadrature set...
          L1(1) = 6.943184420297371E-002
          L1(2) = 6.943184420297371E-002
          L1(3) = 6.943184420297371E-002
          L1(4) = 6.943184420297371E-002
          L1(5) = 6.943184420297371E-002
          L1(6) = 0.330009478207572
          L1(7) = 0.330009478207572
          L1(8) = 0.330009478207572
          L1(9) = 0.330009478207572
          L1(10) = 0.669990521792428
          L1(11) = 0.669990521792428
          L1(12) = 0.669990521792428
          L1(13) = 0.930568155797026
          L1(14) = 0.930568155797026
          ! local coord 1:
          L2(1) = 4.365302387072518E-002
          L2(2) = 0.214742881469342
          L2(3) = 0.465284077898513
          L2(4) = 0.715825274327684
          L2(5) = 0.886915131926301
          L2(6) = 4.651867752656094E-002
          L2(7) = 0.221103222500738
          L2(8) = 0.448887299291690
          L2(9) = 0.623471844265867
          L2(10) = 3.719261778493340E-002
          L2(11) = 0.165004739103786
          L2(12) = 0.292816860422638
          L2(13) = 1.467267513102734E-002
          L2(14) = 5.475916907194637E-002
          ! local coord 2:
          WEIGHT(1) = 1.917346464706755E-002
          WEIGHT(2) = 3.873334126144628E-002
          WEIGHT(3) = 4.603770904527855E-002
          WEIGHT(4) = 3.873334126144628E-002
          WEIGHT(5) = 1.917346464706755E-002
          WEIGHT(6) = 3.799714764789616E-002
          WEIGHT(7) = 7.123562049953998E-002
          WEIGHT(8) = 7.123562049953998E-002
          WEIGHT(9) = 3.799714764789616E-002
          WEIGHT(10) = 2.989084475992800E-002
          WEIGHT(11) = 4.782535161588505E-002
          WEIGHT(12) = 2.989084475992800E-002
          WEIGHT(13) = 6.038050853208200E-003
          WEIGHT(14) = 6.038050853208200E-003
          rsum=SUM(WEIGHT(1:NGI))
          WEIGHT(1:NGI)=WEIGHT(1:NGI)/RSUM
          ! ENDOF IF(NGI.EQ.14) THEN...
       ENDIF

       !
       DO I=1,NGI
          L3(I)=1.0-L1(I)-L2(I)
       END DO
       ! ENDOF IF(.NOT.D3) THEN...
    ENDIF
    !
    RETURN
  END subroutine TRIQUAold
  !
  !



  subroutine shape_triangle_cubic( l1, l2, l3, l4, weight, d3, &
       nloc, ngi, &
       n, nlx, nly, nlz )
    implicit none
    integer, intent( in ) :: nloc, ngi
    real, dimension( ngi ), intent( in ) :: l1, l2, l3, l4, weight
    logical, intent( in ) :: d3
    real, dimension( nloc, ngi ), intent( inout ) :: n, nlx, nly, nlz
    ! Local variables
    logical :: base_order
    integer :: gi, ndim, cv_ele_type_dummy, u_nloc_dummy
    real, dimension( :, : ), allocatable :: cvn_dummy, un_dummy, unlx_dummy, &
         unly_dummy, unlz_dummy
    real, dimension( : ), allocatable :: cvweigh_dummy
    real :: a,b

    !ewrite(3,*)'In shatri d3,nloc=',d3,nloc
    if(nloc.ne.10) then ! wrong element type
       stop 28213
    endif


    ! cubic triangle...
    do gi = 1, ngi
       ! corner nodes...
       n( 1, gi ) = 0.5*( 3. * l1( gi ) - 1. ) * (3. * l1( gi )   -2.) *  l1( gi )
       n( 2, gi ) = 0.5*( 3. * l2( gi ) - 1. ) * (3. * l2( gi )   -2.) *  l2( gi )
       n( 3, gi ) = 0.5*( 3. * l3( gi ) - 1. ) * (3. * l3( gi )   -2.) *  l3( gi )
       ! mid side nodes...
       n( 4, gi ) = (9./2.)*l1( gi )*l2( gi )*( 3. * l1( gi ) - 1. )
       n( 5, gi ) = (9./2.)*l2( gi )*l1( gi )*( 3. * l2( gi ) - 1. )

       n( 6, gi ) = (9./2.)*l2( gi )*l3( gi )*( 3. * l2( gi ) - 1. )
       n( 7, gi ) = (9./2.)*l3( gi )*l2( gi )*( 3. * l3( gi ) - 1. )

       n( 8, gi ) = (9./2.)*l3( gi )*l1( gi )*( 3. * l3( gi ) - 1. )
       n( 9, gi ) = (9./2.)*l1( gi )*l3( gi )*( 3. * l1( gi ) - 1. )
       ! central node...
       n( 10, gi ) = 27.*l1( gi )*l2( gi )*l3( gi )

       ! x-derivative (nb. l1 + l2 + l3  = 1 )
       ! corner nodes...
       nlx( 1, gi ) = 0.5*( 27. * l1( gi )**2  - 18. *  l1( gi ) + 2. )
       nlx( 2, gi ) = 0.0
       nlx( 3, gi ) = 0.5*( 27. * l3( gi )**2  - 18. *  l3( gi ) + 2. )   *  (-1.0)
       ! mid side nodes...
       nlx( 4, gi ) = (9./2.)*(6.*l1( gi )*l2( gi )  - l2( gi ) )
       nlx( 5, gi ) = (9./2.)*l2( gi )*( 3. * l2( gi ) - 1. )

       nlx( 6, gi ) = - (9./2.)*l2( gi )*( 3. * l2( gi ) - 1. )
       nlx( 7, gi ) = (9./2.)*(   -l2(gi)*( 6.*l3(gi) -1. )    )

       nlx( 8, gi ) = -(9./2.)*( l1( gi )*(6.*l3(gi)-1.) + l3(gi)*(3.*l3(gi)-1.)  )
       nlx( 9, gi ) = (9./2.)*(  l3(gi)*(3.*l1(gi)-1.) -l1(gi)*(3.*l1(gi)-1.)  )
       ! central node...
       nlx( 10, gi ) = 27.*l2( gi )*( 1. - 2.*l1(gi)  - l2( gi ) )

       ! y-derivative (nb. l1 + l2 + l3  = 1 )
       ! corner nodes...
       nly( 1, gi ) = 0.0
       nly( 2, gi ) = 0.5*( 27. * l2( gi )**2  - 18. *  l2( gi ) + 2.  )
       nly( 3, gi ) = 0.5*( 27. * l3( gi )**2  - 18. *  l3( gi ) + 2.  )   *  (-1.0)
       ! mid side nodes...
       nly( 4, gi ) = (9./2.)*l1( gi )*( 3. * l1( gi ) - 1. )
       nly( 5, gi ) = (9./2.)*l1( gi )*( 6. * l2( gi ) - 1. )

       nly( 6, gi ) = (9./2.)*( l3( gi )*( 6. * l2( gi ) - 1. ) -l2(gi)*( 3.*l2(gi)-1. )  )
       nly( 7, gi ) = (9./2.)*( -l2( gi )*( 6. * l3( gi ) - 1. ) +l3(gi)*(3.*l3(gi)-1.)  )

       nly( 8, gi ) = -(9./2.)*l1( gi )*( 6. * l3( gi ) - 1. )
       nly( 9, gi ) = -(9./2.)*l1( gi )*( 3. * l1( gi ) - 1. )
       ! central node...
       nly( 10, gi ) = 27.*l1( gi )*( 1. - 2.*l2(gi)  - l1( gi ) )
    end do

  end subroutine shape_triangle_cubic


    	  
!
! 
       SUBROUTINE MATINV(A,N,NMAX,MAT,MAT2,X,B)
! This sub finds the inverse of the matrix A and puts it back in A. 
! MAT, MAT2 & X,B are working vectors. 
       IMPLICIT NONE
       INTEGER N,NMAX
       REAL A(NMAX,NMAX),MAT(N,N),MAT2(N,N),X(N),B(N)
! Local variables
       INTEGER ICOL,IM,JM


         DO IM=1,N
           DO JM=1,N
             MAT(IM,JM)=A(IM,JM)
           END DO
         END DO
! Solve MAT X=B (NB MAT is overwritten).  
       CALL SMLINN_FACTORIZE(MAT,X,B,N,N)
!
       DO ICOL=1,N
!
! Form column ICOL of the inverse. 
         DO IM=1,N
           B(IM)=0.
         END DO
         B(ICOL)=1.0
! Solve MAT X=B (NB MAT is overwritten).  
       CALL SMLINN_SOLVE_LU(MAT,X,B,N,N)
! X contains the column ICOL of inverse
         DO IM=1,N
           MAT2(IM,ICOL)=X(IM)
         END DO 
!
      END DO
!
! Set A to MAT2
         DO IM=1,N
           DO JM=1,N
             A(IM,JM)=MAT2(IM,JM)
           END DO
         END DO
       RETURN
       END SUBROUTINE MATINV
!
!
!     
	  
        SUBROUTINE SMLINN_FACTORIZE(A,X,B,NMX,N)
        IMPLICIT NONE
        INTEGER NMX,N
        REAL A(NMX,NMX),X(NMX),B(NMX)
        REAL R
        INTEGER K,I,J
!     Form X = A^{-1} B
!     Useful subroutine for inverse
!     This sub overwrites the matrix A. 
        DO K=1,N-1
           DO I=K+1,N
              A(I,K)=A(I,K)/A(K,K)
           END DO
           DO J=K+1,N
              DO I=K+1,N
                 A(I,J)=A(I,J) - A(I,K)*A(K,J)
              END DO
           END DO
        END DO
!     
      if(.false.) then
!     Solve L_1 x=b
        DO I=1,N
           R=0.
           DO J=1,I-1
              R=R+A(I,J)*X(J)
           END DO
           X(I)=B(I)-R
        END DO
!     
!     Solve U x=y
        DO I=N,1,-1
           R=0.
           DO J=I+1,N
              R=R+A(I,J)*X(J)
           END DO
           X(I)=(X(I)-R)/A(I,I)
        END DO
      endif
        RETURN
        END SUBROUTINE SMLINN_FACTORIZE
!     
!     
	  
        SUBROUTINE SMLINN_SOLVE_LU(A,X,B,NMX,N)
        IMPLICIT NONE
        INTEGER NMX,N
        REAL A(NMX,NMX),X(NMX),B(NMX)
        REAL R
        INTEGER K,I,J
!     Form X = A^{-1} B
!     Useful subroutine for inverse
!     This sub overwrites the matrix A. 
       if(.false.) then
        DO K=1,N-1
           DO I=K+1,N
              A(I,K)=A(I,K)/A(K,K)
           END DO
           DO J=K+1,N
              DO I=K+1,N
                 A(I,J)=A(I,J) - A(I,K)*A(K,J)
              END DO
           END DO
        END DO
       endif
!     
!     Solve L_1 x=b
        DO I=1,N
           R=0.
           DO J=1,I-1
              R=R+A(I,J)*X(J)
           END DO
           X(I)=B(I)-R
        END DO
!     
!     Solve U x=y
        DO I=N,1,-1
           R=0.
           DO J=I+1,N
              R=R+A(I,J)*X(J)
           END DO
           X(I)=(X(I)-R)/A(I,I)
        END DO
        RETURN
        END SUBROUTINE SMLINN_SOLVE_LU
!     
!       


!     
      SUBROUTINE JACOBL(N,ALPHA,BETA,XJAC)
      IMPLICIT NONE
!     COMPUTES THE GAUSS-LOBATTO COLLOCATION POINTS FOR JACOBI POLYNOMIALS
!     
!     N:       DEGREE OF APPROXIMATION
!     ALPHA:   PARAMETER IN JACOBI WEIGHT
!     BETA:    PARAMETER IN JACOBI WEIGHT
!     
!     XJAC:    OUTPUT ARRAY WITH THE GAUSS-LOBATTO ROOTS
!     THEY ARE ORDERED FROM LARGEST (+1.0) TO SMALLEST (-1.0)
!     
      INTEGER N
      REAL ALPHA,BETA
!      IMPLICIT REAL(A-H,O-Z)
!      REAL XJAC(1)
      REAL XJAC(N+1)
      REAL ALP,BET,RV
      REAL PNP1P,PDNP1P,PNP,PDNP,PNM1P,PDNM1,PNP1M,PDNP1M,PNM,PDNM,PNM1M
      REAL DET,RP,RM,A,B,DTH,CD,SD,CS,SS,X,PNP1,PDNP1,PN,PDN,PNM1,POLY
      REAL PDER,RECSUM,DELX,CSSAVE
      INTEGER NP,NH,J,K,JM,I,NPP
      COMMON /JACPAR/ALP,BET,RV
      INTEGER KSTOP
      DATA KSTOP/10/
      REAL EPS
      DATA EPS/1.0E-12/
      ALP = ALPHA
      BET =BETA
      RV = 1 + ALP
      NP = N+1
!
!  COMPUTE THE PARAMETERS IN THE POLYNOMIAL WHOSE ROOTS ARE DESIRED
!
      CALL JACOBF(NP,PNP1P,PDNP1P,PNP,PDNP,PNM1P,PDNM1,1.0)
      CALL JACOBF(NP,PNP1M,PDNP1M,PNM,PDNM,PNM1M,PDNM1,-1.0)
      DET = PNP*PNM1M-PNM*PNM1P
      RP = -PNP1P
      RM = -PNP1M
      A = (RP*PNM1M-RM*PNM1P)/DET
      B = (RM*PNP-RP*PNM)/DET
!
      XJAC(1) = 1.0
      NH = (N+1)/2
!
!  SET-UP RECURSION RELATION FOR INITIAL GUESS FOR THE ROOTS
!
      DTH = 3.14159265/(2*N+1)
      CD = COS(2.*DTH)
      SD = SIN(2.*DTH)
      CS = COS(DTH)
      SS = SIN(DTH)
!
!  COMPUTE THE FIRST HALF OF THE ROOTS BY POLYNOMIAL DEFLATION
!
      do  J=2,NH! Was loop 39
         X = CS
      do  K=1,KSTOP! Was loop 29
            CALL JACOBF(NP,PNP1,PDNP1,PN,PDN,PNM1,PDNM1,X)
            POLY = PNP1+A*PN+B*PNM1
            PDER = PDNP1+A*PDN+B*PDNM1
            RECSUM = 0.0
            JM = J-1
      do  I=1,JM! Was loop 27
               RECSUM = RECSUM+1.0/(X-XJAC(I))
      end do ! Was loop 27
28          CONTINUE
            DELX = -POLY/(PDER-RECSUM*POLY)
            X = X+DELX
            IF(ABS(DELX) .LT. EPS) GO TO 30
      end do ! Was loop 29
30       CONTINUE
         XJAC(J) = X
         CSSAVE = CS*CD-SS*SD
         SS = CS*SD+SS*CD
         CS = CSSAVE
      end do ! Was loop 39
      XJAC(NP) = -1.0
      NPP = N+2
!
! USE SYMMETRY FOR SECOND HALF OF THE ROOTS
!
      do  I=2,NH! Was loop 49
         XJAC(NPP-I) = -XJAC(I)
      end do ! Was loop 49
      IF(N .NE. 2*(N/2)) RETURN
      XJAC(NH+1) = 0.0
      RETURN
      END SUBROUTINE JACOBL


!     
!     
!     
      REAL FUNCTION LAGRAN(DIFF,LX,INOD,NDNOD,NODPOS)
      IMPLICIT NONE
!     This return the Lagrange poly assocaited with node INOD at point LX
!     If DIFF then send back the value of this poly differentiated. 
      LOGICAL DIFF
      INTEGER INOD,NDNOD
      REAL LX,NODPOS(0:NDNOD-1)
      REAL DENOMI,OVER,OVER1
      INTEGER N,K,I,JJ
!     ewrite(3,*) 'inside lagran'
!     ewrite(3,*) 'DIFF,LX,INOD,NDNOD,NODPOS:',
!     &            DIFF,LX,INOD,NDNOD,NODPOS
!
      N=NDNOD-1
      K=INOD-1
!     
      DENOMI=1.
      do I=0,K-1
         DENOMI=DENOMI*(NODPOS(K)-NODPOS(I))
      END DO
      do I=K+1,N
         DENOMI=DENOMI*(NODPOS(K)-NODPOS(I))
      END DO
!     
      IF(.NOT.DIFF) THEN
         OVER=1.
      do I=0,K-1
            OVER=OVER*(LX-NODPOS(I))
         END DO
      do I=K+1,N
            OVER=OVER*(LX-NODPOS(I))
         END DO
         LAGRAN=OVER/DENOMI
      ELSE
         OVER=0.
      do JJ=0,N
            IF(JJ.NE.K) THEN
               OVER1=1.
      do I=0,K-1
                  IF(JJ.NE.I) OVER1=OVER1*(LX-NODPOS(I))
               END DO
      do I=K+1,N
                  IF(JJ.NE.I) OVER1=OVER1*(LX-NODPOS(I))
               END DO
               OVER=OVER+OVER1
            ENDIF
         END DO
         LAGRAN=OVER/DENOMI
      ENDIF
!     
!     ewrite(3,*) 'FINISHED LAGRAN'
      END
      


      SUBROUTINE LAGROT(WEIT,QUAPOS,NDGI,GETNDP)
!        use RGPTWE_module
      IMPLICIT NONE
!     This computes the weight and points for standard Gaussian quadrature.
!     IF(GETNDP) then get the POSITION OF THE NODES 
!     AND DONT BOTHER WITH THE WEITS.
      INTEGER NDGI
      REAL WEIT(NDGI),QUAPOS(NDGI)
      LOGICAL GETNDP
      LOGICAL WEIGHT
      INTEGER IG
! real function...
      real RGPTWE
!     
      IF(.NOT.GETNDP) THEN
         WEIGHT=.TRUE.
         do IG=1,NDGI
            WEIT(IG)=RGPTWE(IG,NDGI,WEIGHT)
         END DO
!     
         WEIGHT=.FALSE.
         do IG=1,NDGI
            QUAPOS(IG)=RGPTWE(IG,NDGI,WEIGHT)
         END DO
      ELSE
         IF(NDGI.EQ.1) THEN
            QUAPOS(1)=0.
         ELSE
            do IG=1,NDGI
               QUAPOS(IG)= -1+2.*REAL(IG-1)/REAL(NDGI-1)
            END DO
         ENDIF
      ENDIF
      END SUBROUTINE LAGROT




  REAL FUNCTION RGPTWE(IG,ND,WEIGHT)
    IMPLICIT NONE
    !     NB If WEIGHT is TRUE in function RGPTWE then return the Gauss-pt weight 
    !     else return the Gauss-pt. 
    !     NB there are ND Gauss points we are looking for either the 
    !     weight or the x-coord of the IG'th Gauss point. 
    INTEGER IG,ND
    LOGICAL WEIGHT

    IF(WEIGHT) THEN
       GO TO (10,20,30,40,50,60,70,80,90,100) ND
       !     +++++++++++++++++++++++++++++++
       !     For N=1 +++++++++++++++++++++++
10     CONTINUE
       RGPTWE=2.0
       GO TO 1000
       !     For N=2 +++++++++++++++++++++++
20     CONTINUE
       RGPTWE=1.0
       GO TO 1000
       ! For N=3 +++++++++++++++++++++++
30     CONTINUE
       GO TO (11,12,11) IG
11     RGPTWE= 0.555555555555556
       GO TO 1000
12     RGPTWE= 0.888888888888889
       GO TO 1000
       ! For N=4 +++++++++++++++++++++++
40     CONTINUE
       GO TO (21,22,22,21) IG
21     RGPTWE= 0.347854845137454
       GO TO 1000
22     RGPTWE= 0.652145154862546
       GO TO 1000
       ! For N=5 +++++++++++++++++++++++
50     CONTINUE
       GO TO (31,32,33,32,31) IG
31     RGPTWE= 0.236926885056189
       GO TO 1000
32     RGPTWE= 0.478628670499366
       GO TO 1000
33     RGPTWE= 0.568888888888889
       GO TO 1000
       ! For N=6 +++++++++++++++++++++++
60     CONTINUE
       GO TO (41,42,43,43,42,41) IG
41     RGPTWE= 0.171324492379170
       GO TO 1000
42     RGPTWE= 0.360761573048139
       GO TO 1000
43     RGPTWE= 0.467913934572691
       GO TO 1000
       ! For N=7 +++++++++++++++++++++++
70     CONTINUE
       GO TO (51,52,53,54,53,52,51) IG
51     RGPTWE= 0.129484966168870
       GO TO 1000
52     RGPTWE= 0.279705391489277
       GO TO 1000
53     RGPTWE= 0.381830050505119
       GO TO 1000
54     RGPTWE= 0.417959183673469
       GO TO 1000
       ! For N=8 +++++++++++++++++++++++
80     CONTINUE
       GO TO (61,62,63,64,64,63,62,61) IG
61     RGPTWE= 0.101228536290376
       GO TO 1000
62     RGPTWE= 0.222381034453374
       GO TO 1000
63     RGPTWE= 0.313706645877877
       GO TO 1000
64     RGPTWE= 0.362683783378362
       GO TO 1000
       ! For N=9 +++++++++++++++++++++++
90     CONTINUE
       GO TO (71,72,73,74,75,74,73,72,71) IG
71     RGPTWE= 0.081274388361574
       GO TO 1000
72     RGPTWE= 0.180648160694857
       GO TO 1000
73     RGPTWE= 0.260610696402935
       GO TO 1000
74     RGPTWE= 0.312347077040003
       GO TO 1000
75     RGPTWE= 0.330239355001260
       GO TO 1000
       ! For N=10 +++++++++++++++++++++++
100    CONTINUE
       GO TO (81,82,83,84,85,85,84,83,82,81) IG
81     RGPTWE= 0.066671344308688
       GO TO 1000
82     RGPTWE= 0.149451349150581
       GO TO 1000
83     RGPTWE= 0.219086362515982
       GO TO 1000
84     RGPTWE= 0.269266719309996
       GO TO 1000
85     RGPTWE= 0.295524224714753
       !
1000   CONTINUE
    ELSE
       GO TO (210,220,230,240,250,260,270,280,290,200) ND
       ! +++++++++++++++++++++++++++++++
       ! For N=1 +++++++++++++++++++++++ THE GAUSS POINTS...
210    CONTINUE
       RGPTWE=0.0
       GO TO 2000
       ! For N=2 +++++++++++++++++++++++
220    CONTINUE
       RGPTWE= 0.577350269189626
       GO TO 2000
       ! For N=3 +++++++++++++++++++++++
230    CONTINUE
       GO TO (211,212,211) IG
211    RGPTWE= 0.774596669241483
       GO TO 2000
212    RGPTWE= 0.0
       GO TO 2000
       ! For N=4 +++++++++++++++++++++++
240    CONTINUE
       GO TO (221,222,222,221) IG
221    RGPTWE= 0.861136311594953
       GO TO 2000
222    RGPTWE= 0.339981043584856
       GO TO 2000
       ! For N=5 +++++++++++++++++++++++
250    CONTINUE
       GO TO (231,232,233,232,231) IG
231    RGPTWE= 0.906179845938664
       GO TO 2000
232    RGPTWE= 0.538469310105683
       GO TO 2000
233    RGPTWE= 0.0
       GO TO 2000
       ! For N=6 +++++++++++++++++++++++
260    CONTINUE
       GO TO (241,242,243,243,242,241) IG
241    RGPTWE= 0.932469514203152
       GO TO 2000
242    RGPTWE= 0.661209386466265
       GO TO 2000
243    RGPTWE= 0.238619186083197
       GO TO 2000
       ! For N=7 +++++++++++++++++++++++
270    CONTINUE
       GO TO (251,252,253,254,253,252,251) IG
251    RGPTWE= 0.949107912342759
       GO TO 2000
252    RGPTWE= 0.741531185599394
       GO TO 2000
253    RGPTWE= 0.405845151377397
       GO TO 2000
254    RGPTWE= 0.0
       GO TO 2000
       ! For N=8 +++++++++++++++++++++++
280    CONTINUE
       GO TO (261,262,263,264,264,263,262,261) IG
261    RGPTWE= 0.960289856497536
       GO TO 2000
262    RGPTWE= 0.796666477413627
       GO TO 2000
263    RGPTWE= 0.525532409916329
       GO TO 2000
264    RGPTWE= 0.183434642495650
       GO TO 2000
       ! For N=9 +++++++++++++++++++++++
290    CONTINUE
       GO TO (271,272,273,274,275,274,273,272,271) IG
271    RGPTWE= 0.968160239507626
       GO TO 2000
272    RGPTWE= 0.836031107326636
       GO TO 2000
273    RGPTWE= 0.613371432700590
       GO TO 2000
274    RGPTWE= 0.324253423403809
       GO TO 2000
275    RGPTWE= 0.0
       GO TO 2000
       ! For N=10 +++++++++++++++++++++++
200    CONTINUE
       GO TO (281,282,283,284,285,285,284,283,282,281) IG
281    RGPTWE= 0.973906528517172
       GO TO 2000
282    RGPTWE= 0.865063366688985
       GO TO 2000
283    RGPTWE= 0.679409568299024
       GO TO 2000
284    RGPTWE= 0.433395394129247
       GO TO 2000
285    RGPTWE= 0.148874338981631
       !
2000   CONTINUE
       IF(IG.LE.INT((ND/2)+0.1)) RGPTWE=-RGPTWE
    ENDIF
  END FUNCTION RGPTWE




      SUBROUTINE JACOBF(N,POLY,PDER,POLYM1,PDERM1,POLYM2,PDERM2,X)
      IMPLICIT NONE
!     
!     COMPUTES THE JACOBI POLYNOMIAL (POLY) AND ITS DERIVATIVE
!     (PDER) OF DEGREE  N  AT  X
!     
      INTEGER N
      REAL APB,POLY,PDER,POLYM1,PDERM1,POLYM2,PDERM2,X
!     IMPLICIT REAL(A-H,O-Z)
      COMMON /JACPAR/ALP,BET,RV
      REAL ALP,BET,RV,POLYLST,PDERLST,A1,A2,B3,A3,A4
      REAL POLYN,PDERN,PSAVE,PDSAVE
      INTEGER K
      APB = ALP+BET
      POLY = 1.0
      PDER = 0.0
      IF(N .EQ. 0) RETURN
      POLYLST = POLY
      PDERLST = PDER
      POLY = RV * X
      PDER = RV
      IF(N .EQ. 1) RETURN
      do K=2,N
         A1 = 2.*K*(K+APB)*(2.*K+APB-2.)
         A2 = (2.*K+APB-1.)*(ALP**2-BET**2)
         B3 = (2.*K+APB-2.)
         A3 = B3*(B3+1.)*(B3+2.)
         A4 = 2.*(K+ALP-1)*(K+BET-1.)*(2.*K+APB)
         POLYN = ((A2+A3*X)*POLY-A4*POLYLST)*A1
         PDERN = ((A2+A3*X)*PDER-A4*PDERLST+A3*POLY)*A1
         PSAVE = POLYLST
         PDSAVE = PDERLST
         POLYLST = POLY
         POLY = POLYN
         PDERLST = PDER
         PDER = PDERN
      END DO
      POLYM1 = POLYLST
      PDERM1 = PDERLST
      POLYM2 = PSAVE
      PDERM2 = PDSAVE
      RETURN
      END SUBROUTINE JACOBF





