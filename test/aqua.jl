import Aqua
import BiophysViabilityModel
using Test: @testset

@testset "aqua" begin
    Aqua.test_all(PhageTree; ambiguities = false)
end
