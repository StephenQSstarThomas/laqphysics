! Atomic units. Native kernels, no external Fortran libraries required.
module helium_kernels
  use iso_c_binding
  implicit none
  real(c_double), parameter :: pi=acos(-1.0_c_double)
  complex(c_double_complex), parameter :: zi=(0.0_c_double,1.0_c_double)
contains
  subroutine fft(a,n,sgn)
    integer,intent(in)::n,sgn
    complex(c_double_complex),intent(inout)::a(n)
    integer::i,j,m,span,k,half
    complex(c_double_complex)::v,w,wm,u
    j=1
    do i=1,n
      if(j>i) then
        v=a(j);a(j)=a(i);a(i)=v
      end if
      m=n/2
      do while(m>=1.and.j>m)
        j=j-m;m=m/2
      end do
      j=j+m
    end do
    span=2
    do while(span<=n)
      half=span/2;wm=exp(zi*real(sgn,c_double)*2*pi/real(span,c_double))
      do k=1,n,span
        w=1
        do j=0,half-1
          u=a(k+j);v=w*a(k+j+half)
          a(k+j)=u+v;a(k+j+half)=u-v;w=w*wm
        end do
      end do
      span=span*2
    end do
    if(sgn==1)a=a/real(n,c_double)
  end subroutine

  subroutine fft2(a,n,sgn)
    integer,intent(in)::n,sgn
    complex(c_double_complex),intent(inout)::a(n,n)
    complex(c_double_complex)::row(n)
    integer::j
    !$omp parallel do private(j) schedule(static)
    do j=1,n
      call fft(a(:,j),n,sgn)
    end do
    !$omp end parallel do
    !$omp parallel do private(j,row) schedule(static)
    do j=1,n
      row=a(j,:);call fft(row,n,sgn);a(j,:)=row
    end do
    !$omp end parallel do
  end subroutine

  ! Each stage can be called separately so the debugger stores every module output.
  subroutine split2_stage(n,psi,v,p,dt,avec,stage) bind(C)
    integer(c_int),value::n,stage
    real(c_double),value::dt,avec
    complex(c_double_complex),intent(inout)::psi(n,n)
    complex(c_double_complex),intent(in)::v(n,n)
    real(c_double),intent(in)::p(n)
    integer::i,j
    if(stage==1.or.stage==3)then
      !$omp parallel do collapse(2) private(i,j) schedule(static)
      do j=1,n
        do i=1,n
          psi(i,j)=psi(i,j)*exp(-zi*dt*v(i,j)/2)
        end do
      end do
      !$omp end parallel do
    else if(stage==2)then
      call fft2(psi,n,-1)
      !$omp parallel do collapse(2) private(i,j) schedule(static)
      do j=1,n
        do i=1,n
          psi(i,j)=psi(i,j)*exp(-zi*dt*((p(i)+avec)**2+(p(j)+avec)**2)/2)
        end do
      end do
      !$omp end parallel do
      call fft2(psi,n,1)
    end if
  end subroutine

  subroutine split2_cached(n,psi,vhalf,pphase) bind(C)
    integer(c_int),value::n
    complex(c_double_complex),intent(inout)::psi(n,n)
    complex(c_double_complex),intent(in)::vhalf(n,n),pphase(n)
    integer::i,j
    !$omp parallel do collapse(2) private(i,j) schedule(static)
    do j=1,n
      do i=1,n
        psi(i,j)=psi(i,j)*vhalf(i,j)
      end do
    end do
    !$omp end parallel do
    call fft2(psi,n,-1)
    !$omp parallel do collapse(2) private(i,j) schedule(static)
    do j=1,n
      do i=1,n
        psi(i,j)=psi(i,j)*pphase(i)*pphase(j)
      end do
    end do
    !$omp end parallel do
    call fft2(psi,n,1)
    !$omp parallel do collapse(2) private(i,j) schedule(static)
    do j=1,n
      do i=1,n
        psi(i,j)=psi(i,j)*vhalf(i,j)
      end do
    end do
    !$omp end parallel do
  end subroutine

  ! Tensor radial Hamiltonian in normalized FE-DVR coordinates and product Y_lm channels.
  ! T: CSR radial kinetic matrix (zero-based indices at the C interface).
  ! Angular Coulomb CSR edges carry lambda and radial multipole factors.
  ! Dipole edges carry electron index, Cartesian angular coefficient, l(l+1) difference.
  subroutine tensor_apply(n,nc,nt,ptr,col,tv,diag,nv,vptr,vc,vl,vcoef,rad, &
                          nd,dptr,dc,de,dcoef,ldiff,r,field,velocity,psi,out) bind(C)
    integer(c_int),value::n,nc,nt,nv,nd,velocity
    integer(c_int),intent(in)::ptr(n+1),col(nt),vptr(nc+1),vc(nv),vl(nv)
    integer(c_int),intent(in)::dptr(nc+1),dc(nd),de(nd)
    complex(c_double_complex),intent(in)::tv(nt),diag(n,n,nc),vcoef(nv),rad(n,n,*)
    complex(c_double_complex),intent(in)::dcoef(3,nd),r(n),psi(n,n,nc)
    real(c_double),intent(in)::field(3),ldiff(nd)
    complex(c_double_complex),intent(out)::out(n,n,nc)
    integer::i,j,c,k,b,s,axis
    complex(c_double_complex)::val,coef,part
    !$omp parallel do collapse(2) private(c,j,i,k,b,s,axis,val,coef,part) schedule(static)
    do c=1,nc
      do j=1,n
        do i=1,n
          val=diag(i,j,c)*psi(i,j,c)
          do k=ptr(i)+1,ptr(i+1)
            b=col(k)+1;val=val+tv(k)*psi(b,j,c)
          end do
          do k=ptr(j)+1,ptr(j+1)
            b=col(k)+1;val=val+tv(k)*psi(i,b,c)
          end do
          do k=vptr(c)+1,vptr(c+1)
            val=val+vcoef(k)*rad(i,j,vl(k)+1)*psi(i,j,vc(k)+1)
          end do
          do k=dptr(c)+1,dptr(c+1)
            coef=sum(field*dcoef(:,k));if(abs(coef)<1d-30)cycle
            s=dc(k)+1
            if(velocity==0)then
              if(de(k)==1)then
                val=val+coef*r(i)*psi(i,j,s)
              else
                val=val+coef*r(j)*psi(i,j,s)
              end if
            else
              if(de(k)==1)then
                part=0
                if(ldiff(k)/=0)part=ldiff(k)/(2*r(i))*psi(i,j,s)
                do axis=ptr(i)+1,ptr(i+1)
                  b=col(axis)+1;part=part+tv(axis)*(r(b)-r(i))*psi(b,j,s)
                end do
              else
                part=0
                if(ldiff(k)/=0)part=ldiff(k)/(2*r(j))*psi(i,j,s)
                do axis=ptr(j)+1,ptr(j+1)
                  b=col(axis)+1;part=part+tv(axis)*(r(b)-r(j))*psi(i,b,s)
                end do
              end if
              val=val+zi*coef*part
            end if
          end do
          if(velocity==1)val=val+sum(field**2)*psi(i,j,c)
          out(i,j,c)=val
        end do
      end do
    end do
    !$omp end parallel do
  end subroutine
end module
