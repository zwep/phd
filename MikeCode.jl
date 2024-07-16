#=
MikeCode:
- Julia version: 
- Author: bugger
- Date: 2020-10-19
=#


### De functie die wordt aangeroepen
print("Alooe\n")
####
#### meas_abs_B1 B1 veld als 1D Array
#### Wavepars is een 5 element Array
#### Maxorder is tot welke orde
#### Xcoords, Ycoords, Zcoords zijn allemaal 1d arrays met alleen die coordinaten (Een tensor product van de 3 beschrijft de ruimte)
#### Wirecords L bij 3 Array waarbij L de lengte van de draad is
function fitB1_Spacy(meas_abs_B1, Wavepars, maxorder, Xcoords, Ycoords, Zcoords, wire_coords; r_annulus = 3*(Xcoords[2] - Xcoords[1]))
    ω = Wavepars[1]
    μ₀ = Wavepars[2]
    ϵ₀ = Wavepars[3]
    ϵᵣ = Wavepars[4]
    σ  = Wavepars[5]

    ka = sqrt(ϵᵣ*ϵ₀*μ₀*ω^2 + 1im*σ*ω*μ₀)
    M = maxorder
    #print("$cylorders")
    cylorders = TM.cylBesselFunctionOrders(M);
    sporders = TM.sphBesselFunctionOrders(M);
    orders = [cylorders;sporders]
    #print("$sporders")
    ### orders: een 1Darray waarin 2-tuples staan die de ordes van de SPACY beschrijven, in dezelfde volgorde als de kolommen van de SPACY matrix

    #orders = sporders
    basis_size = size(orders,1)

    #determine the Size of the voxels.
    VoxelSize = [Xcoords[2] - Xcoords[1]; Ycoords[2] - Ycoords[1]; Zcoords[2] - Zcoords[1]];

    ## Coordinates Generation
    ϕˢᵖʰ, θˢᵖʰ, rˢᵖʰ = TM.cart2sph(Xcoords,Ycoords,Zcoords);
    ϕᶜʸˡ, rᶜʸˡ = TM.cart2cyl(Xcoords,Ycoords,Zcoords);

    ϕˢᵖʰ_wire, θˢᵖʰ_wire, rˢᵖʰ_wire, ϕᶜʸˡ_wire, rᶜʸˡ_wire, wire_space = TM.generateWireCoordinates(wire_coords, Xcoords, Ycoords, Zcoords);
    wire_z = wire_space[:,3];

    ### Matrix to calculate electric fields at the wire (a.k.a -2i*d/dx - 2*d/dy)
    B = generate_D(ϕˢᵖʰ_wire, θˢᵖʰ_wire, rˢᵖʰ_wire, ϕᶜʸˡ_wire, rᶜʸˡ_wire, M, ka)


    print("Generating Mask with radius: $r_annulus m \n")
    NoV = ( size(Xcoords,1), size(Ycoords,1) , size(Zcoords,1))
    Mask = generate_Mask_annulus(reshape(meas_abs_B1, NoV), Xcoords, Ycoords, Zcoords, wire_space, r_annulus)
    MaskIndices = findall(x -> x > 0, Mask[:])
    print("\n \n \n")

    ## Filtering fitting data
    print("Filtering Coordinates and Data:")
    MaskIndices = findall(x -> x > 0, Mask[:])
    rᶜʸˡ_filtered = rᶜʸˡ[MaskIndices]
    ϕᶜʸˡ_filtered = ϕᶜʸˡ[MaskIndices]

    rˢᵖʰ_filtered = rˢᵖʰ[MaskIndices]
    θˢᵖʰ_filtered = θˢᵖʰ[MaskIndices]
    ϕˢᵖʰ_filtered = ϕˢᵖʰ[MaskIndices]
    print("\n \n \n")
    meas_abs_B1_filtered = meas_abs_B1[MaskIndices]
    nData = size(rᶜʸˡ_filtered,1)
    print("$nData  points \n \n \n")

    ## Generate The SPACY matrix for the grid with the hole at the Artifact
    print("Precomputing A \n")
    cylbf = TM.cylBesselFunction(rᶜʸˡ_filtered[:],ϕᶜʸˡ_filtered[:],ka,M);
    spbf  = TM.sphBesselFunction(rˢᵖʰ_filtered[:],θˢᵖʰ_filtered[:],ϕˢᵖʰ_filtered[:],ka,M);
    A = convert.(ComplexF64, spbf);
    @show size(A)
    @show TM.cond(A)
    print("\n \n \n")


    print("Running Simple Spacy Solver: You may now get a beer. \n")
    Spacy_Coeff = TM.LinearAlgebra.pinv(A)*meas_abs_B1_filtered


    ### Generating full grid without the annulus in the middle
    cylbf = TM.cylBesselFunction(rᶜʸˡ[:],ϕᶜʸˡ[:],ka,M);
    spbf  = TM.sphBesselFunction(rˢᵖʰ[:],θˢᵖʰ[:],ϕˢᵖʰ[:],ka,M);
    A = convert.(ComplexF64, spbf);
    SpaCy_Field = A*Spacy_Coeff;
    ### Computing Electric Fields at the wire positions
    Ez_wire = B*Spacy_Coeff;

    return Spacy_Coeff, SpaCy_Field, Mask, Ez_wire

end


function sphBesselFunction(rSp,θ,ϕ,ka,M)
    output = Array{Complex,2}(undef,length(rSp),(M+1)^2)
    Pnm    = Array{AbstractFloat,1}(undef,length(θ))
    jn     = Array{Complex,1}(undef,length(θ))
    Ynm    = Array{Complex,1}(undef,length(θ))
    idx    = 0
    rSp    = ka.*rSp
    for n in 0:M
        @. jn = besselj(n+0.5,rSp)*sqrt(pi/(2.0*rSp))
        for m in -n:n
            @. Pnm = associatedLegendrePoly(cos(θ),n,m) # This function was so bugged at first
            @. Ynm = sqrt((2*n+1)*factorial(n-m)/(4*pi*factorial(n+m)))*Pnm*exp(1im*m*ϕ)
            idx += 1
            output[:,idx] = jn.*Ynm
        end
    end

    return output

end


function cylBesselFunction(rCyl,PH,ka,M)
    # construct cylindrical Bessel-Fourier functions for ALL orders -M to M
    output = Array{Complex,2}(undef,prod(size(rCyl)),2*M+1)
    for order in -M:M
        @. output[:,order+M+1] = besselj(order, ka*rCyl)*exp(1im*order*PH)
        #output[:,order+M+1] = output[:,order+M+1]./norm(output[:,order+M+1])
    end
    return output
end