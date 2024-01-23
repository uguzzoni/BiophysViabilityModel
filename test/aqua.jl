import Aqua
import BiophysViabilityModel
using Test: @testset

@testset "aqua" begin
    Aqua.test_all(BiophysViabilityModel; ambiguities = false)
end
