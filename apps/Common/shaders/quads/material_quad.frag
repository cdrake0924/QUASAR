layout(location = 0) out vec4 FragColor;
layout(location = 1) out vec4 FragNormal;
layout(location = 2) out uvec4 FragIDs;

in VertexData {
    flat uint DrawID;
    vec3 TexCoord3D;
    vec2 ExpandedUV;
    vec3 FragPos;
} fsIn;

// Material
uniform struct Material {
    vec4 baseColor;
    vec4 baseColorFactor;

    int alphaMode;
    float maskThreshold;

    bool hasBaseColorMap; // use color map
    bool hasAlphaMap; // use alpha map

    // Material textures
    sampler2D baseColorMap; // 0
    sampler2D alphaMap; // 1
} material;

void main() {
    vec4 color;
    vec2 uv = fsIn.TexCoord3D.xy / fsIn.TexCoord3D.z;
    if (material.hasBaseColorMap) {
        color = texture(material.baseColorMap, uv) * material.baseColorFactor;
    }
    else {
        color = material.baseColor * material.baseColorFactor;
    }

    float alpha = (material.alphaMode == ALPHA_OPAQUE) ? 1.0 : color.a;
    if (material.hasAlphaMap) {
        alpha = texture(material.alphaMap, uv).r;
    }
    if (alpha < material.maskThreshold)
        discard;

    FragColor = vec4(color.rgb, alpha);
    // FragNormal = vec4(normalize(fsIn.Normal), 1.0);
    FragIDs = uvec4(fsIn.DrawID, gl_PrimitiveID, 0.0, 1.0);
}
