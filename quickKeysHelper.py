"""
ULTRA-OPTIMIZED Layer-aware transform keying utilities.
Maximum performance by bypassing DG evaluation and reading curves directly.

FIXED: Properly handles muted layers - they are completely ignored in delta calculation.
"""
import maya.cmds as cmds
import maya.api.OpenMaya as om2
import maya.api.OpenMayaAnim as oma


# ============================================================================
# Layer Stack Cache - Build Once, Reuse
# ============================================================================


class LayerStackCache:
    """
    Cache layer hierarchy, weights, modes, and animation curves.
    Eliminates repeated queries during baking operations.
    """
    def __init__(self, node, attrs, target_layer):
        self.node = node
        self.attrs = attrs
        self.target_layer = target_layer

        # Cache node's rotation order
        rot_order_idx = cmds.getAttr(f"{node}.rotateOrder")
        rotation_orders = ['xyz', 'yzx', 'zxy', 'xzy', 'yxz', 'zyx']
        self.rotation_order = rotation_orders[rot_order_idx]

        # Build layer hierarchy
        self.all_layers = self._getAnimationLayers()
        self.target_idx = self.all_layers.index(target_layer) if target_layer in self.all_layers else len(self.all_layers)

        # FIXED: Include ALL layers EXCEPT target (both below AND above)
        # When keying AnimLayer2, we need to account for AnimLayer1 above it too
        self.layers_to_composite = [l for l in self.all_layers if l != target_layer]

        # Cache layer properties for ALL layers (including target for its mode)
        self.layer_is_override = {}
        self.layer_scale_mode = {}
        self.layer_rotation_mode = {}

        for layer in self.all_layers:
            if layer == 'BaseAnimation':
                self.layer_is_override[layer] = True
                self.layer_scale_mode[layer] = 'additive'
                self.layer_rotation_mode[layer] = 'component'
            else:
                self.layer_is_override[layer] = cmds.getAttr(f"{layer}.override") == 1
                mode = cmds.getAttr(f"{layer}.scaleAccumulationMode")
                self.layer_scale_mode[layer] = 'multiply' if mode == 1 else 'additive'
                rot_mode = cmds.getAttr(f"{layer}.rotationAccumulationMode")
                self.layer_rotation_mode[layer] = 'layer' if rot_mode == 1 else 'component'

        # Cache animation curves for each attr on each layer (EXCEPT target)
        self.curves_cache = self._buildCurvesCache()

    def _getAnimationLayers(self):
        """Get all animation layers in order from bottom to top."""
        all_layers = cmds.ls(type='animLayer')
        if not all_layers:
            return ['BaseAnimation']

        base_layer = 'BaseAnimation'
        ordered_layers = [base_layer]

        # Get children of BaseAnimation (these are in order)
        base_children = cmds.animLayer(base_layer, query=True, children=True)
        if base_children:
            ordered_layers.extend(base_children)

        # Handle nested layers (if any layer has children)
        for layer in base_children if base_children else []:
            children = cmds.animLayer(layer, query=True, children=True)
            if children:
                # Insert children after their parent
                parent_idx = ordered_layers.index(layer)
                for child in children:
                    ordered_layers.insert(parent_idx + 1, child)

        return ordered_layers

    def _buildCurvesCache(self):
        """
        Build cache of animation curves for all attrs on all layers EXCEPT target.
        CRITICAL: Muted layers are completely excluded from the cache.

        Returns: {attr: [(layer, MFnAnimCurve or None, weight_attr, mute_attr), ...]}
        """
        curves = {attr: [] for attr in self.attrs}

        for attr in self.attrs:
            # FIXED: Iterate over layers_to_composite (excludes target)
            for layer in self.layers_to_composite:
                # CRITICAL: Check if layer is muted BEFORE adding to cache
                if layer != 'BaseAnimation':
                    try:
                        is_muted = cmds.getAttr(f"{layer}.mute")
                        if is_muted:
                            print(f"    Skipping muted layer '{layer}' for {attr}")
                            continue
                    except:
                        pass

                curve_fn = None
                weight_attr = None
                mute_attr = None

                if layer == 'BaseAnimation':
                    curve_fn = self._findBaseAnimCurve(attr)
                else:
                    curve_fn = self._findLayerAnimCurve(attr, layer)
                    weight_attr = f"{layer}.weight"
                    mute_attr = f"{layer}.mute"

                curves[attr].append((layer, curve_fn, weight_attr, mute_attr))

        return curves

    def _findBaseAnimCurve(self, attr):
        """Find animation curve on base animation layer."""
        plug_name = f"{self.node}.{attr}"

        try:
            # Find outermost blend node
            blend_nodes = cmds.listConnections(plug_name, source=True, destination=False,
                                             type='animBlendNodeAdditiveDL')
            if not blend_nodes:
                blend_nodes = cmds.listConnections(plug_name, source=True, destination=False,
                                                type='animBlendNodeAdditiveRotation')
            if not blend_nodes:
                blend_nodes = cmds.listConnections(plug_name, source=True, destination=False,
                                                type='animBlendNodeAdditiveScale')

            if blend_nodes:
                # Traverse inputA chain until we find a curve
                current_node = blend_nodes[0]

                while True:
                    # Determine correct input name based on attribute
                    input_name = self._getInputAName(attr, current_node)

                    try:
                        inputA_conn = cmds.listConnections(f'{current_node}.{input_name}',
                                                          source=True, destination=False)
                        if inputA_conn:
                            node_type = cmds.nodeType(inputA_conn[0])

                            # If it's another blend node, keep traversing
                            if 'Blend' in node_type:
                                current_node = inputA_conn[0]
                                continue

                            # If it's a curve, return it
                            if 'animCurve' in node_type:
                                selList = om2.MSelectionList()
                                selList.add(inputA_conn[0])
                                return oma.MFnAnimCurve(selList.getDependNode(0))
                    except:
                        pass

                    # No more connections
                    break
            else:
                # No blend nodes - direct connection
                selList = om2.MSelectionList()
                selList.add(plug_name)
                plug = selList.getPlug(0)

                if oma.MAnimUtil.isAnimated(plug):
                    curves = oma.MAnimUtil.findAnimation(plug)
                    if curves:
                        return oma.MFnAnimCurve(curves[0])
        except:
            pass

        return None

    def _getInputAName(self, attr, blend_node):
        """Get correct inputA name based on attribute and blend type."""
        blend_type = cmds.nodeType(blend_node)

        if blend_type == "animBlendNodeAdditiveRotation":
            if attr == "rotateX":
                return "inputAX"
            elif attr == "rotateY":
                return "inputAY"
            elif attr == "rotateZ":
                return "inputAZ"

        # For translate/scale, use inputA
        return "inputA"

    def _findLayerAnimCurve(self, attr, layer):
        """Find animation curve on a specific layer by traversing blend nodes."""
        plug_name = f"{self.node}.{attr}"

        try:
            # Find blend nodes
            blend_nodes = cmds.listConnections(plug_name, source=True, destination=False,
                                             type='animBlendNodeAdditiveDL')
            if not blend_nodes:
                blend_nodes = cmds.listConnections(plug_name, source=True, destination=False,
                                                type='animBlendNodeAdditiveRotation')
            if not blend_nodes:
                blend_nodes = cmds.listConnections(plug_name, source=True, destination=False,
                                                type='animBlendNodeAdditiveScale')
            if not blend_nodes:
                return None

            blend_node = find_blend_node_for_layer(plug_name, layer)
            if not blend_node:
                return None

            # Get correct inputB name based on attribute
            input_name = self._getInputBName(attr, blend_node)

            try:
                curves = cmds.listConnections(f"{blend_node}.{input_name}",
                                             source=True, destination=False,
                                             type='animCurve')
                if curves:
                    selList = om2.MSelectionList()
                    selList.add(curves[0])
                    return oma.MFnAnimCurve(selList.getDependNode(0))
            except:
                pass
        except:
            pass

        return None

    def _getInputBName(self, attr, blend_node):
        """Get correct inputB name based on attribute and blend type."""
        blend_type = cmds.nodeType(blend_node)

        if blend_type == "animBlendNodeAdditiveRotation":
            if attr == "rotateX":
                return "inputBX"
            elif attr == "rotateY":
                return "inputBY"
            elif attr == "rotateZ":
                return "inputBZ"

        return "inputB"


# ============================================================================
# Fast Curve Evaluation
# ============================================================================

def evaluateCurvesDirectly(cache, times):
    """
    Evaluate animation curves directly and compose layers manually.
    This is THE key optimization - bypasses all DG evaluation.

    CRITICAL FIX: Properly skips muted layers by reading mute attribute without time parameter.

    Args:
        cache (LayerStackCache): Pre-built cache of curves and layer info
        times (list): Frame numbers to evaluate

    Returns:
        dict: {attr: [values]} - composite values at each time
    """
    scale_attrs = {'scaleX', 'scaleY', 'scaleZ'}
    results = {attr: [] for attr in cache.attrs}

    # Pre-create MTime objects
    mtimes = [om2.MTime(t, om2.MTime.uiUnit()) for t in times]

    # For each time point
    for mtime in mtimes:
        time_val = mtime.value

        # For each attribute
        for attr in cache.attrs:
            is_scale = attr in scale_attrs
            composite_value = 1.0 if is_scale else 0.0  # Identity for scale
            last_override_value = None

            # Compose layers from bottom to top
            for layer, curve_fn, weight_attr, mute_attr in cache.curves_cache[attr]:
                # CRITICAL FIX: Check if layer is muted (only for non-base layers)
                # The mute attribute is NOT animated, so DON'T pass time parameter!
                # Passing time=time_val was causing the mute check to fail.
                if mute_attr:
                    try:
                        is_muted = cmds.getAttr(mute_attr)  # Read current mute state
                        if is_muted > 0.5:
                            # Layer is muted - skip it entirely in composition
                            continue
                    except:
                        # If we can't read mute state, assume not muted and continue
                        pass

                # Get layer weight (only for non-base layers)
                layer_weight = 1.0
                if weight_attr:
                    try:
                        layer_weight = cmds.getAttr(weight_attr, time=time_val)
                        if layer_weight < 0.0001:
                            continue  # Skip zero-weight layers
                    except:
                        pass

                # Evaluate curve at this time
                curve_value = 0.0
                if curve_fn:
                    try:
                        curve_value = curve_fn.evaluate(mtime)
                        if 'rotate' in attr:
                            curve_value = om2.MAngle(curve_value).asDegrees()
                    except:
                        curve_value = 0.0
                else:
                    # No curve - use identity value
                    if is_scale:
                        # For scale attributes, identity depends on context
                        if layer == 'BaseAnimation':
                            # BaseAnimation is override - scale default is 1.0
                            curve_value = 1.0
                        elif cache.layer_scale_mode[layer] == 'multiply':
                            # Layer multiply mode - identity is 1.0
                            curve_value = 1.0
                        else:
                            # Layer add mode - identity is 0.0
                            curve_value = 0.0
                    else:
                        # Translate/Rotate - identity is always 0.0
                        curve_value = 0.0

                # Apply to composite based on layer mode
                if cache.layer_is_override[layer]:
                    # Override layer - replace previous value
                    last_override_value = curve_value
                    composite_value = curve_value
                else:
                    # Additive layer
                    if is_scale and cache.layer_scale_mode[layer] == 'multiply':
                        # Multiply mode for scale
                        composite_value *= (1.0 + (curve_value - 1.0) * layer_weight)
                    else:
                        # Additive mode
                        composite_value += curve_value * layer_weight

            results[attr].append(composite_value)

    return results


def evaluateCurvesDirectlyFallback(cache, times):
    """
    Fallback method using plug evaluation with layer muting.
    Faster than original but not as fast as direct curve evaluation.
    Use this if direct curve reading is unreliable.
    """
    # Mute target layer and above
    layers_to_restore = []
    for layer in cache.all_layers[cache.target_idx:]:
        if layer != 'BaseAnimation' and cmds.objExists(layer):
            was_muted = cmds.getAttr(f"{layer}.mute")
            if not was_muted:
                cmds.setAttr(f"{layer}.mute", 1)
                layers_to_restore.append(layer)

    try:
        # Get plugs once
        plugs = {}
        selList = om2.MSelectionList()
        for attr in cache.attrs:
            plug_name = f"{cache.node}.{attr}"
            try:
                selList.clear()
                selList.add(plug_name)
                plugs[attr] = selList.getPlug(0)
            except:
                plugs[attr] = None

        # Pre-create contexts
        contexts = [om2.MDGContext(om2.MTime(t, om2.MTime.uiUnit())) for t in times]

        # Evaluate all
        results = {attr: [] for attr in cache.attrs}
        for ctx in contexts:
            for attr in cache.attrs:
                plug = plugs[attr]
                if plug:
                    try:
                        val = plug.asDouble(ctx)
                        # Workaround: If scale is exactly 0.0 when querying composite values,
                        # it's likely a Maya bug with animation layers. Default to 1.0 (identity).
                        # Note: Legitimate near-zero scale (0.001, etc) will be preserved.
                        if attr.startswith('scale') and val == 0.0:
                            val = 1.0
                    except:
                        # Default value depends on attribute type
                        # Scale defaults to 1.0 (identity), others to 0.0
                        val = 1.0 if attr.startswith('scale') else 0.0
                else:
                    val = 1.0 if attr.startswith('scale') else 0.0
                results[attr].append(val)

        return results

    finally:
        # Restore mute states
        for layer in layers_to_restore:
            cmds.setAttr(f"{layer}.mute", 0)


# ============================================================================
# Fast Delta Calculation
# ============================================================================

def calculateDeltaValuesFast(cache, times, world_values, use_direct_eval=True):
    """
    Ultra-fast delta calculation using direct curve evaluation.

    Args:
        cache (LayerStackCache): Pre-built layer/curve cache
        times (list): Frame numbers
        world_values (dict): Target world values {attr: [values]}
        use_direct_eval (bool): If True, use direct curve reading (fastest)

    Returns:
        dict: Delta values {attr: [values]}
    """
    """Ultra-fast delta calculation using direct curve evaluation."""

    # DEBUG: Print what world values we're trying to match
    print(f"\n=== calculateDeltaValuesFast DEBUG ===")
    print(f"Target node: {cache.node}")
    print(f"Target layer: {cache.target_layer}")
    print(f"World values being used (should be SOURCE's values):")
    for attr in ['translateX', 'translateY', 'translateZ']:
        if attr in world_values:
            print(f"  {attr}: {world_values[attr][0]:.6f}")
    # In calculateDeltaValuesFast(), expand the debug:
    print(f"World values being used (should be SOURCE's values):")
    for attr in ['translateX', 'translateY', 'translateZ', 'rotateX', 'rotateY', 'rotateZ', 'scaleX', 'scaleY', 'scaleZ']:
        if attr in world_values:
            print(f"  {attr}: {world_values[attr][0]:.6f}")
    # Override layer - just return world values
    if cache.layer_is_override[cache.target_layer] or cache.target_layer == 'BaseAnimation':
        return world_values

    # Check if any layers being composited use "layer" rotation accumulation mode
    # If so, we MUST use Maya's evaluation to get proper quaternion-based composition
    any_layer_uses_quat_rotation = False
    rotation_attrs = {'rotateX', 'rotateY', 'rotateZ'}
    if rotation_attrs.issubset(set(cache.attrs)):
        for layer in cache.layers_to_composite:
            if cache.layer_rotation_mode[layer] == 'layer':
                any_layer_uses_quat_rotation = True
                break

    # Get composite values from layers below target
    # IMPORTANT: If any layer uses quaternion rotation mode, we must use Maya's evaluation
    # because direct curve reading won't respect the quaternion composition
    if use_direct_eval and not any_layer_uses_quat_rotation:
        try:
            composite_values = evaluateCurvesDirectly(cache, times)
        except Exception as e:
            print(f"Direct curve evaluation failed: {e}, falling back to plug evaluation")
            composite_values = evaluateCurvesDirectlyFallback(cache, times)
    else:
        if any_layer_uses_quat_rotation:
            print(f"Using Maya evaluation for composite because layers use quaternion rotation mode")
        composite_values = evaluateCurvesDirectlyFallback(cache, times)

    # Calculate deltas with rotation unwrapping
    scale_attrs = {'scaleX', 'scaleY', 'scaleZ'}
    rotation_attrs = {'rotateX', 'rotateY', 'rotateZ'}
    delta_values = {}

    # Check if we need to use quaternion math for rotations
    rotation_mode = cache.layer_rotation_mode[cache.target_layer]
    use_quat_rotation = rotation_mode == 'layer' and rotation_attrs.issubset(set(cache.attrs))

    # Process rotation attributes together if using quaternion mode
    if use_quat_rotation:
        print(f"\n=== QUATERNION ROTATION MODE DEBUG ===")
        print(f"Target layer: {cache.target_layer}")
        print(f"Rotation order: {cache.rotation_order}")

        # Calculate quaternion-based rotation deltas
        delta_rx_list = []
        delta_ry_list = []
        delta_rz_list = []

        # IMPORTANT: evaluateCurvesDirectlyFallback returns rotation values in RADIANS
        # but our quaternion functions expect DEGREES. Must convert!
        for i in range(len(times)):
            world_rx = world_values['rotateX'][i]
            world_ry = world_values['rotateY'][i]
            world_rz = world_values['rotateZ'][i]

            # Convert composite rotations from radians to degrees
            composite_rx_rad = composite_values['rotateX'][i]
            composite_ry_rad = composite_values['rotateY'][i]
            composite_rz_rad = composite_values['rotateZ'][i]

            composite_rx = om2.MAngle(composite_rx_rad, om2.MAngle.kRadians).asDegrees()
            composite_ry = om2.MAngle(composite_ry_rad, om2.MAngle.kRadians).asDegrees()
            composite_rz = om2.MAngle(composite_rz_rad, om2.MAngle.kRadians).asDegrees()

            if i == 0:  # Debug first frame
                print(f"\nFrame {times[i]}:")
                print(f"  World rotation (deg): ({world_rx:.2f}, {world_ry:.2f}, {world_rz:.2f})")
                print(f"  Composite rotation (rad): ({composite_rx_rad:.4f}, {composite_ry_rad:.4f}, {composite_rz_rad:.4f})")
                print(f"  Composite rotation (deg): ({composite_rx:.2f}, {composite_ry:.2f}, {composite_rz:.2f})")

            delta_rx, delta_ry, delta_rz = calculateRotationDeltaQuaternion(
                world_rx, world_ry, world_rz,
                composite_rx, composite_ry, composite_rz,
                cache.rotation_order
            )

            if i == 0:  # Debug first frame
                print(f"  Delta rotation (deg): ({delta_rx:.2f}, {delta_ry:.2f}, {delta_rz:.2f})")

            # Unwrap deltas to be continuous
            if delta_rx_list:
                delta_rx = unwrapAngle(delta_rx, delta_rx_list[-1])
                delta_ry = unwrapAngle(delta_ry, delta_ry_list[-1])
                delta_rz = unwrapAngle(delta_rz, delta_rz_list[-1])

            delta_rx_list.append(delta_rx)
            delta_ry_list.append(delta_ry)
            delta_rz_list.append(delta_rz)

        delta_values['rotateX'] = delta_rx_list
        delta_values['rotateY'] = delta_ry_list
        delta_values['rotateZ'] = delta_rz_list
        print("=== END QUATERNION DEBUG ===\n")

    # Process non-rotation attributes (or rotations if using component mode)
    for attr in cache.attrs:
        # Skip rotation attrs if we already processed them with quaternion math
        if use_quat_rotation and attr in rotation_attrs:
            continue

        is_scale = attr in scale_attrs
        scale_mode = cache.layer_scale_mode[cache.target_layer]

        # Unwrap composite rotations to match world rotations
        composite_vals = composite_values[attr][:]
        if attr.startswith('rotate'):
            for i in range(1, len(composite_vals)):
                composite_vals[i] = unwrapAngle(composite_vals[i], composite_vals[i-1])
            composite_values[attr] = composite_vals

        deltas = []
        for i in range(len(times)):
            world_val = world_values[attr][i]
            composite = composite_vals[i]

            if is_scale and scale_mode == 'multiply':
                delta = world_val / composite if abs(composite) > 0.0001 else world_val
            else:
                delta = world_val - composite
                if attr.startswith('rotate') and deltas:
                    delta = unwrapAngle(delta, deltas[-1])

            deltas.append(delta)

        delta_values[attr] = deltas

    # DEBUG: Print scale delta values
    print(f"\n=== SCALE DELTA DEBUG ===")
    for attr in ['scaleX', 'scaleY', 'scaleZ']:
        if attr in delta_values:
            print(f"  {attr} deltas (first 3): {delta_values[attr][:3]}")
    print("=== END SCALE DEBUG ===\n")

    return delta_values


# ============================================================================
# Fast Matrix Operations
# ============================================================================

def getWorldMatricesFast(node, times):
    """
    Get world matrices using pure API - no cmds calls.

    Args:
        node (str): Node name
        times (list): Frame numbers

    Returns:
        dict: {time: matrix_as_list}
    """

    # DEBUG: Verify which node we're reading
    print(f"\n=== getWorldMatricesFast DEBUG ===")
    print(f"Reading matrices from node: {node}")

    # Get the DAG path once
    selList = om2.MSelectionList()
    selList.add(node)
    dag_path = selList.getDagPath(0)

    print(f"DAG path: {dag_path.fullPathName()}")


    # Get the DAG path once
    selList = om2.MSelectionList()
    selList.add(node)
    dag_path = selList.getDagPath(0)

    # Get the worldMatrix plug
    fn_transform = om2.MFnTransform(dag_path)
    world_matrix_plug = fn_transform.findPlug("worldMatrix", False)
    world_matrix_plug = world_matrix_plug.elementByLogicalIndex(0)

    matrices = {}

    # Evaluate at each time
    for time in times:
        mtime = om2.MTime(time, om2.MTime.uiUnit())
        ctx = om2.MDGContext(mtime)

        # Get matrix object
        matrix_obj = world_matrix_plug.asMObject(ctx)
        matrix_data = om2.MFnMatrixData(matrix_obj)
        matrix = matrix_data.matrix()

        # Convert to flat list
        matrices[time] = [matrix.getElement(r, c) for r in range(4) for c in range(4)]

    return matrices


def eulerToQuaternion(rx, ry, rz, rotation_order='xyz'):
    """
    Convert euler angles to quaternion using Maya API.

    Args:
        rx, ry, rz (float): Rotation angles in degrees
        rotation_order (str): Rotation order

    Returns:
        om2.MQuaternion: Quaternion
    """
    rotation_order_map = {
        'xyz': om2.MEulerRotation.kXYZ,
        'yzx': om2.MEulerRotation.kYZX,
        'zxy': om2.MEulerRotation.kZXY,
        'xzy': om2.MEulerRotation.kXZY,
        'yxz': om2.MEulerRotation.kYXZ,
        'zyx': om2.MEulerRotation.kZYX
    }
    rot_order = rotation_order_map.get(rotation_order.lower(), om2.MEulerRotation.kXYZ)

    # Convert degrees to radians
    rx_rad = om2.MAngle(rx, om2.MAngle.kDegrees).asRadians()
    ry_rad = om2.MAngle(ry, om2.MAngle.kDegrees).asRadians()
    rz_rad = om2.MAngle(rz, om2.MAngle.kDegrees).asRadians()

    euler = om2.MEulerRotation(rx_rad, ry_rad, rz_rad, rot_order)
    return euler.asQuaternion()


def quaternionToEuler(quat, rotation_order='xyz'):
    """
    Convert quaternion to euler angles using Maya API.

    Args:
        quat (om2.MQuaternion): Quaternion
        rotation_order (str): Rotation order

    Returns:
        tuple: (rx, ry, rz) in degrees
    """
    rotation_order_map = {
        'xyz': om2.MEulerRotation.kXYZ,
        'yzx': om2.MEulerRotation.kYZX,
        'zxy': om2.MEulerRotation.kZXY,
        'xzy': om2.MEulerRotation.kXZY,
        'yxz': om2.MEulerRotation.kYXZ,
        'zyx': om2.MEulerRotation.kZYX
    }
    rot_order = rotation_order_map.get(rotation_order.lower(), om2.MEulerRotation.kXYZ)

    euler = quat.asEulerRotation()
    euler = euler.reorder(rot_order)

    rx = om2.MAngle(euler.x).asDegrees()
    ry = om2.MAngle(euler.y).asDegrees()
    rz = om2.MAngle(euler.z).asDegrees()

    return (rx, ry, rz)


def calculateRotationDeltaQuaternion(world_rx, world_ry, world_rz,
                                     composite_rx, composite_ry, composite_rz,
                                     rotation_order='xyz'):
    """
    Calculate rotation delta using quaternion math.

    For additive layers with "layer" rotation accumulation mode:
        world_quat = delta_quat * composite_quat (Maya's quaternion composition order)
        delta_quat = world_quat * inverse(composite_quat)

    Args:
        world_rx, world_ry, world_rz (float): Target world rotation in degrees
        composite_rx, composite_ry, composite_rz (float): Composite rotation from layers below in degrees
        rotation_order (str): Rotation order

    Returns:
        tuple: (delta_rx, delta_ry, delta_rz) in degrees
    """
    # Convert euler to quaternions
    world_quat = eulerToQuaternion(world_rx, world_ry, world_rz, rotation_order)
    composite_quat = eulerToQuaternion(composite_rx, composite_ry, composite_rz, rotation_order)

    # Calculate delta: delta_quat = world_quat * inverse(composite_quat)
    # Note: Maya uses right-to-left quaternion multiplication
    composite_quat_inv = composite_quat.inverse()
    delta_quat = world_quat * composite_quat_inv

    # Convert back to euler
    delta_rx, delta_ry, delta_rz = quaternionToEuler(delta_quat, rotation_order)

    return (delta_rx, delta_ry, delta_rz)


def decomposeMatricesBatch(matrices, times, rotation_order='xyz', euler_filter=False, node=None):
    """
    Decompose multiple matrices in batch.
    Pre-allocates result dict for speed.

    Args:
        matrices (dict): {time: 16-element matrix list}
        times (list): Frame numbers
        rotation_order (str): Rotation order
        euler_filter (bool): If True, apply euler unwrapping to prevent flips. Default: False
        node (str): Node name - if provided, scale will be queried directly from attributes
                    instead of extracted from matrix (workaround for animation layer bug)

    Returns:
        dict: {attr: [values]} for all 9 transform attributes
    """
    # Pre-allocate results
    results = {
        'translateX': [], 'translateY': [], 'translateZ': [],
        'rotateX': [], 'rotateY': [], 'rotateZ': [],
        'scaleX': [], 'scaleY': [], 'scaleZ': []
    }

    # Rotation order mapping
    rotation_order_map = {
        'xyz': om2.MEulerRotation.kXYZ,
        'yzx': om2.MEulerRotation.kYZX,
        'zxy': om2.MEulerRotation.kZXY,
        'xzy': om2.MEulerRotation.kXZY,
        'yxz': om2.MEulerRotation.kYXZ,
        'zyx': om2.MEulerRotation.kZYX
    }
    rot_order = rotation_order_map.get(rotation_order.lower(), om2.MEulerRotation.kXYZ)

    # Decompose each matrix
    prev_rotation = None
    for time in times:
        matrix_list = matrices[time]
        m_matrix = om2.MMatrix(matrix_list)
        m_transform = om2.MTransformationMatrix(m_matrix)

        # Translation
        translation = m_transform.translation(om2.MSpace.kWorld)
        results['translateX'].append(translation.x)
        results['translateY'].append(translation.y)
        results['translateZ'].append(translation.z)

        # Rotation with euler unwrapping
        rotation = m_transform.rotation().reorder(rot_order)

        # Convert to degrees
        rx = om2.MAngle(rotation.x).asDegrees()
        ry = om2.MAngle(rotation.y).asDegrees()
        rz = om2.MAngle(rotation.z).asDegrees()

        # Unwrap rotations to prevent flips (if enabled)
        # If we have a previous rotation, adjust current to be continuous
        if euler_filter and prev_rotation is not None:
            rx = unwrapAngle(rx, prev_rotation[0])
            ry = unwrapAngle(ry, prev_rotation[1])
            rz = unwrapAngle(rz, prev_rotation[2])

        results['rotateX'].append(rx)
        results['rotateY'].append(ry)
        results['rotateZ'].append(rz)

        if euler_filter:
            prev_rotation = (rx, ry, rz)

        # Scale - will be populated later (see below loop)

        if euler_filter:
            results['rotateX'], results['rotateY'], results['rotateZ'] = eulerFilter(
                results['rotateX'], results['rotateY'], results['rotateZ'])

    # Query scale attributes directly if node is provided
    # This is a workaround for Maya bug where worldMatrix contains zero-length basis vectors
    # when animation layers with multiply scale mode are present but scale isn't keyed
    if node:
        # Get scale plugs
        scale_plugs = {}
        selList = om2.MSelectionList()
        for attr in ['scaleX', 'scaleY', 'scaleZ']:
            plug_name = f"{node}.{attr}"
            try:
                selList.clear()
                selList.add(plug_name)
                scale_plugs[attr] = selList.getPlug(0)
            except:
                scale_plugs[attr] = None

        # Query scale at each time
        for time in times:
            ctx = om2.MDGContext(om2.MTime(time, om2.MTime.uiUnit()))
            for attr in ['scaleX', 'scaleY', 'scaleZ']:
                plug = scale_plugs[attr]
                if plug:
                    try:
                        val = plug.asDouble(ctx)
                    except:
                        val = 1.0  # Default scale
                else:
                    val = 1.0
                results[attr].append(val)
    else:
        # Fallback: extract from matrix (may return 0,0,0 with animation layers)
        # Re-read matrices to get scale
        import math
        for time in times:
            matrix_list = matrices[time]
            m_matrix = om2.MMatrix(matrix_list)

            # Scale is the length of the basis vectors
            sx = math.sqrt(m_matrix[0] * m_matrix[0] + m_matrix[1] * m_matrix[1] + m_matrix[2] * m_matrix[2])
            sy = math.sqrt(m_matrix[4] * m_matrix[4] + m_matrix[5] * m_matrix[5] + m_matrix[6] * m_matrix[6])
            sz = math.sqrt(m_matrix[8] * m_matrix[8] + m_matrix[9] * m_matrix[9] + m_matrix[10] * m_matrix[10])

            results['scaleX'].append(sx)
            results['scaleY'].append(sy)
            results['scaleZ'].append(sz)

    return results


def unwrapAngle(current, previous):
    """
    Unwrap an angle to be continuous with the previous angle.
    Prevents 360 degree flips in rotation curves.

    Args:
        current (float): Current angle in degrees
        previous (float): Previous angle in degrees

    Returns:
        float: Unwrapped angle that's continuous with previous
    """
    # Find the difference
    diff = current - previous

    # Adjust by multiples of 360 to find the smallest jump
    while diff > 180:
        current -= 360
        diff = current - previous

    while diff < -180:
        current += 360
        diff = current - previous

    return current

import numpy as np

def eulerFilter(rot_x, rot_y, rot_z):
    """
    Ultra-optimized euler filter using NumPy vectorization.
    Handles gimbal lock and rotation wrapping to prevent flips.
    """
    if len(rot_x) < 2:
        return rot_x, rot_y, rot_z

    # Convert to numpy arrays for vectorized operations
    rx = np.array(rot_x, dtype=np.float64)
    ry = np.array(rot_y, dtype=np.float64)
    rz = np.array(rot_z, dtype=np.float64)

    n = len(rx)

    # Pre-allocate output arrays
    filtered_x = np.empty(n, dtype=np.float64)
    filtered_y = np.empty(n, dtype=np.float64)
    filtered_z = np.empty(n, dtype=np.float64)

    # Initialize with first frame
    filtered_x[0] = rx[0]
    filtered_y[0] = ry[0]
    filtered_z[0] = rz[0]

    # Pre-compute offset combinations (3^3 = 27 combinations)
    offsets = np.array([0, 360, -360], dtype=np.float64)
    ox, oy, oz = np.meshgrid(offsets, offsets, offsets, indexing='ij')
    offset_combos = np.stack([ox.ravel(), oy.ravel(), oz.ravel()], axis=1)  # Shape: (27, 3)

    # Pre-compute gimbal flip bases (4 variations)
    gimbal_bases = np.array([
        [180, 180, 180],
        [-180, 180, -180],
        [180, -180, 180],
        [-180, -180, -180]
    ], dtype=np.float64)

    for i in range(1, n):
        prev = np.array([filtered_x[i-1], filtered_y[i-1], filtered_z[i-1]])
        current = np.array([rx[i], ry[i], rz[i]])

        # Generate simple candidates: current + all offset combinations
        simple_candidates = current + offset_combos  # Shape: (27, 3)

        # Compute distances (Manhattan distance for speed)
        distances = np.abs(simple_candidates - prev).sum(axis=1)

        # Find best simple candidate
        best_idx = np.argmin(distances)
        best = simple_candidates[best_idx]
        best_dist = distances[best_idx]

        # Only check gimbal flips if distance is large
        if best_dist > 60:
            # Generate gimbal candidates
            # For each gimbal base, apply transformations and offsets
            gimbal_current = np.empty((4, 3), dtype=np.float64)
            gimbal_current[0] = [current[0] + 180, 180 - current[1], current[2] + 180]
            gimbal_current[1] = [current[0] - 180, 180 - current[1], current[2] - 180]
            gimbal_current[2] = [current[0] + 180, -180 - current[1], current[2] + 180]
            gimbal_current[3] = [current[0] - 180, -180 - current[1], current[2] - 180]

            # Add offsets to each gimbal candidate (4 * 27 = 108 candidates)
            gimbal_candidates = (gimbal_current[:, np.newaxis, :] + offset_combos[np.newaxis, :, :])
            gimbal_candidates = gimbal_candidates.reshape(-1, 3)  # Shape: (108, 3)

            # Combine all candidates
            all_candidates = np.vstack([simple_candidates, gimbal_candidates])

            # Recompute distances for all candidates
            distances = np.abs(all_candidates - prev).sum(axis=1)
            best_idx = np.argmin(distances)
            best = all_candidates[best_idx]

        filtered_x[i] = best[0]
        filtered_y[i] = best[1]
        filtered_z[i] = best[2]

    return filtered_x.tolist(), filtered_y.tolist(), filtered_z.tolist()

# ============================================================================
# Optimized High-Level Functions
# ============================================================================

def setTransformKeysOnLayerFast(node, transform_data, times, layer=None,
                                attrs=None, rotation_order='xyz', euler_filter=False, source_node=None):
    """
    Ultra-optimized version of setTransformKeysOnLayer.
    Uses caching and direct curve evaluation.

    Args:
        node (str): Target node to set keys on
        source_node (str): Source node that transform_data came from (for scale query workaround)
        euler_filter (bool): If True, apply euler unwrapping to prevent flips. Default: False
    """
    if layer is None:
        # from . import getActiveAnimationLayer
        layer = getActiveAnimationLayer()

    if layer != 'BaseAnimation' and not cmds.objExists(layer):
        raise RuntimeError(f"Layer '{layer}' does not exist")

    # Convert transform data to attribute values
    first_val = list(transform_data.values())[0]
    is_matrix = isinstance(first_val, (list, tuple)) and len(first_val) == 16

    if is_matrix:
        # Use source_node for scale query (workaround for animation layer bug)
        # If source_node not provided, fall back to extracting from matrix
        scale_query_node = source_node if source_node else None
        attr_values = decomposeMatricesBatch(transform_data, times, rotation_order, euler_filter, node=scale_query_node)
        if attrs:
            attr_values = {k: v for k, v in attr_values.items() if k in attrs}
    else:
        # Already have attribute dicts
        attr_values = {}
        for time in times:
            attr_dict = transform_data.get(time)
            if attr_dict is None:
                raise ValueError(f"No transform data for time {time}")
            for attr, value in attr_dict.items():
                if attrs is None or attr in attrs:
                    if attr not in attr_values:
                        attr_values[attr] = []
                    attr_values[attr].append(value)

    # Build cache once
    cache = LayerStackCache(node, list(attr_values.keys()), layer)

    # Calculate deltas using fast method
    final_values = calculateDeltaValuesFast(cache, times, attr_values, use_direct_eval=True)

    # Ensure target layer is unmuted
    if layer != 'BaseAnimation' and cmds.objExists(layer):
        cmds.setAttr(f"{layer}.mute", 0)

    # Build flat lists for quickKeys
    attr_list = [f"{node}.{attr}" for attr in final_values.keys()]
    flat_values = []
    for time_idx in range(len(times)):
        for attr in final_values.keys():
            flat_values.append(final_values[attr][time_idx])

    # Call quickKeys plugin
    cmds.quickKeys(attr_list, f=times, v=flat_values, l=layer)

    print(f"Set {len(times)} keyframes on {len(attr_values)} attributes "
          f"for '{node}' on layer '{layer}'")


def bakeTransformToLayerFast(source_node, target_node, start_time, end_time,
                             layer=None, sample_by=1, attrs=None, euler_filter=False):
    """
    Ultra-optimized baking using pure API for matrix queries.

    Args:
        euler_filter (bool): If True, apply euler unwrapping to prevent flips. Default: False
    """
    if layer is None:
        # from . import getActiveAnimationLayer
        layer = getActiveAnimationLayer()

    # Generate time list
    times = []
    current_time = start_time
    while current_time <= end_time:
        times.append(current_time)
        current_time += sample_by

    # Get all matrices using fast API method
    matrices = getWorldMatricesFast(source_node, times)

    # Get rotation order (one cmds call is acceptable during setup)
    rot_order = cmds.getAttr(f"{source_node}.rotateOrder")  # Fix: use source
    rot_order_names = ['xyz', 'yzx', 'zxy', 'xzy', 'yxz', 'zyx']
    rot_order_str = rot_order_names[rot_order]

    # Set keys using ultra-fast method
    setTransformKeysOnLayerFast(target_node, matrices, times, layer,
                                attrs=attrs, rotation_order=rot_order_str,
                                euler_filter=euler_filter, source_node=source_node)

    print(f"Baked {len(times)} frames from '{source_node}' to '{target_node}' "
          f"on layer '{layer}'")


# ============================================================================
# Backward Compatibility - Keep Old Function Names
# ============================================================================

def setTransformKeysOnLayer(*args, **kwargs):
    """Backward compatible wrapper - redirects to fast version."""
    return setTransformKeysOnLayerFast(*args, **kwargs)


def bakeTransformToLayer(*args, **kwargs):
    """Backward compatible wrapper - redirects to fast version."""
    return bakeTransformToLayerFast(*args, **kwargs)


def getActiveAnimationLayer():
    """Get the currently active animation layer."""
    root_layer = cmds.animLayer(query=True, root=True)
    active_layers = cmds.animLayer(query=True, bestAnimLayer=True)
    if active_layers:
        return active_layers[0]
    return 'BaseAnimation'


# Add this to the END of quickKeysHelper.py

# ============================================================================
# Simple User-Facing Interface
# ============================================================================

def cloneTransform(source=None, target=None, start_frame=None, end_frame=None,
                   layer='BaseAnimation', sample_by=1, create_locator=True, euler_filter=False):
    """
    Clone one transform to another in world space.

    Simple interface for everyday use.

    Args:
        source (str): Source object name. If None, uses first selection.
        target (str): Target object name. If None, uses second selection or creates locator.
        start_frame (int): Start frame. If None, uses timeline start.
        end_frame (int): End frame. If None, uses timeline end.
        layer (str): Animation layer to key on. Default: 'BaseAnimation'
        sample_by (int): Sample every N frames. Default: 1 (every frame)
        create_locator (bool): If True and no target, creates a locator. Default: True
        euler_filter (bool): If True, apply euler unwrapping to prevent rotation flips. Default: False

    Returns:
        str: Target object name

    Examples:
        # Select one object, create locator, bake to it:
        cloneTransform()

        # Select source then target, bake between them:
        cloneTransform()

        # Specify objects by name:
        cloneTransform('pSphere1', 'locator1')

        # Custom frame range:
        cloneTransform(start_frame=10, end_frame=100)

        # Bake to a specific layer:
        cloneTransform(layer='MyAnimLayer')

        # Sample every 5 frames (faster):
        cloneTransform(sample_by=5)

        # Enable euler filtering:
        cloneTransform(euler_filter=True)
    """
    # Get source and target from selection if not provided
    if source is None or target is None:
        sel = cmds.ls(selection=True, transforms=True)

        if not sel:
            raise ValueError("No objects selected. Select source (and optionally target).")

        if source is None:
            source = sel[0]

        if target is None:
            if len(sel) >= 2:
                target = sel[1]
            elif create_locator:
                # Create locator at source's position
                target = cmds.spaceLocator(name=f"{source}_baked")[0]
                print(f"Created locator: {target}")
            else:
                raise ValueError("No target specified and only one object selected.")

    # Get frame range from timeline if not provided
    if start_frame is None:
        start_frame = int(cmds.playbackOptions(query=True, minTime=True))
    if end_frame is None:
        end_frame = int(cmds.playbackOptions(query=True, maxTime=True))

    # Ensure target is on the layer (required for non-BaseAnimation layers)
    if layer != 'BaseAnimation':
        if not cmds.objExists(layer):
            cmds.animLayer(layer)
            print(f"Created animation layer: {layer}")

        # Add target to layer
        cmds.select(target, replace=True)
        cmds.animLayer(layer, edit=True, addSelectedObjects=True)
        cmds.select(clear=True)

    # Do the bake!
    bakeTransformToLayerFast(
        source_node=source,
        target_node=target,
        start_time=start_frame,
        end_time=end_frame,
        layer=layer,
        sample_by=sample_by,
        euler_filter=euler_filter
    )

    return target


"""
Diagnostic tool to trace quickKeys delta calculation.
Add this to quickKeysHelper.py to debug baking issues.
"""
import maya.cmds as cmds
import maya.api.OpenMaya as om2


def diagnose_bake_calculation(node, target_layer, frame=1):
    """
    Trace the delta calculation for a single frame to see what's wrong.

    Args:
        node: The target node (e.g., 'target')
        target_layer: The layer being baked to (e.g., 'AnimLayer2')
        frame: Frame to diagnose (default: 1)
    """
    print("\n" + "="*80)
    print(f"DIAGNOSTIC: Baking '{node}' to layer '{target_layer}' at frame {frame}")
    print("="*80)

    # Set current time
    cmds.currentTime(frame)

    # Get all layers
    all_layers = cmds.ls(type='animLayer')
    if not all_layers:
        all_layers = []

    # Build layer hierarchy
    base_layer = 'BaseAnimation'
    ordered_layers = [base_layer]
    remaining = [l for l in all_layers if l != base_layer]

    while remaining:
        for layer in remaining[:]:
            parent_conn = cmds.listConnections(f"{layer}.parentLayer",
                                              source=True, destination=False)
            parent = parent_conn[0] if parent_conn else base_layer
            if parent in ordered_layers:
                parent_idx = ordered_layers.index(parent)
                ordered_layers.insert(parent_idx + 1, layer)
                remaining.remove(layer)

    print(f"\nLayer Hierarchy (bottom to top):")
    for i, layer in enumerate(ordered_layers):
        marker = " <-- TARGET" if layer == target_layer else ""
        print(f"  {i}: {layer}{marker}")

    # FIXED: Composite all layers EXCEPT target (both below and above)
    layers_to_composite = [l for l in ordered_layers if l != target_layer]
    print(f"\nLayers to composite (all except target): {layers_to_composite}")

    # For each transform attribute, trace the values
    attrs = ['translateX', 'translateY', 'translateZ',
             'rotateX', 'rotateY', 'rotateZ',
             'scaleX', 'scaleY', 'scaleZ']

    for attr in attrs:
        print(f"\n--- {attr} ---")

        # 1. World value (what we want target to be)
        world_val = cmds.getAttr(f"{node}.{attr}")
        print(f"  World Value (target): {world_val:.6f}")

        # 2. Trace through each layer (all except target)
        print(f"  Composition breakdown:")

        composite = 1.0 if 'scale' in attr else 0.0

        for layer in layers_to_composite:
            # Check if layer is muted
            is_muted = False
            if layer != 'BaseAnimation':
                try:
                    is_muted = cmds.getAttr(f"{layer}.mute")
                except:
                    pass

            if is_muted:
                print(f"    {layer}: MUTED (skipped)")
                continue

            # Get layer weight
            layer_weight = 1.0
            if layer != 'BaseAnimation':
                try:
                    layer_weight = cmds.getAttr(f"{layer}.weight", time=frame)
                except:
                    pass

            # Find the curve value on this layer
            curve_val = get_layer_curve_value(node, attr, layer, frame)

            # Get layer mode
            is_override = True
            scale_mode = 'additive'
            if layer != 'BaseAnimation':
                try:
                    is_override = cmds.getAttr(f"{layer}.override") == 1
                    mode_val = cmds.getAttr(f"{layer}.scaleAccumulationMode")
                    scale_mode = 'multiply' if mode_val == 1 else 'additive'
                except:
                    pass

            # Compose
            if is_override:
                composite = curve_val
                print(f"    {layer}: {curve_val:.6f} (OVERRIDE) -> composite = {composite:.6f}")
            else:
                if 'scale' in attr and scale_mode == 'multiply':
                    old_composite = composite
                    composite *= (1.0 + (curve_val - 1.0) * layer_weight)
                    print(f"    {layer}: {curve_val:.6f} * weight={layer_weight:.2f} (MULTIPLY) -> composite = {composite:.6f}")
                else:
                    old_composite = composite
                    composite += curve_val * layer_weight
                    print(f"    {layer}: {curve_val:.6f} * weight={layer_weight:.2f} (ADD) -> composite = {composite:.6f}")

        # 3. Calculate delta
        is_scale = 'scale' in attr
        if target_layer != 'BaseAnimation':
            try:
                target_scale_mode = cmds.getAttr(f"{target_layer}.scaleAccumulationMode")
                target_scale_mode = 'multiply' if target_scale_mode == 1 else 'additive'
            except:
                target_scale_mode = 'additive'
        else:
            target_scale_mode = 'additive'

        if is_scale and target_scale_mode == 'multiply':
            if abs(composite) < 0.0001:
                delta = world_val
            else:
                delta = world_val / composite
            print(f"  Delta (multiply mode): {world_val:.6f} / {composite:.6f} = {delta:.6f}")
        else:
            delta = world_val - composite
            if attr.startswith('rotate') and deltas:
                delta = unwrapAngle(delta, deltas[-1])
            print(f"  Delta (additive mode): {world_val:.6f} - {composite:.6f} = {delta:.6f}")

        # 4. Verify: if we key this delta on target layer, do we get world value?
        print(f"  Verification: {composite:.6f} + {delta:.6f} = {composite + delta:.6f} (should be {world_val:.6f})")

    print("\n" + "="*80)


def get_layer_curve_value(node, attr, layer, frame):
    """Get the curve value for a specific attribute on a specific layer."""
    plug_name = f"{node}.{attr}"

    # Determine correct input names based on attribute
    if attr == 'rotateX':
        inputA_name, inputB_name = 'inputAX', 'inputBX'
    elif attr == 'rotateY':
        inputA_name, inputB_name = 'inputAY', 'inputBY'
    elif attr == 'rotateZ':
        inputA_name, inputB_name = 'inputAZ', 'inputBZ'
    else:
        inputA_name, inputB_name = 'inputA', 'inputB'

    if layer == 'BaseAnimation':
        # Find outermost blend
        blend_nodes = cmds.listConnections(plug_name, source=True, destination=False,
                                         type='animBlendNodeAdditiveDL')
        if not blend_nodes:
            blend_nodes = cmds.listConnections(plug_name, source=True, destination=False,
                                            type='animBlendNodeAdditiveRotation')
        if not blend_nodes:
            blend_nodes = cmds.listConnections(plug_name, source=True, destination=False,
                                            type='animBlendNodeAdditiveScale')

        if blend_nodes:
            # Traverse inputA chain
            current = blend_nodes[0]
            while True:
                try:
                    conn = cmds.listConnections(f'{current}.{inputA_name}', source=True)
                    if not conn:
                        break
                    if 'Blend' in cmds.nodeType(conn[0]):
                        current = conn[0]
                    elif 'animCurve' in cmds.nodeType(conn[0]):
                        return cmds.getAttr(f'{conn[0]}.output', time=frame)
                    else:
                        break
                except:
                    break

        return 1.0 if 'scale' in attr else 0.0

    else:
        # Find blend for this layer
        blend_node = find_blend_node_for_layer(plug_name, layer)
        if not blend_node:
            return 1.0 if 'scale' in attr else 0.0

        try:
            curves = cmds.listConnections(f"{blend_node}.{inputB_name}",
                                         source=True, type='animCurve')
            if curves:
                return cmds.getAttr(f'{curves[0]}.output', time=frame)
        except:
            pass

        return 1.0 if 'scale' in attr else 0.0


def find_blend_node_for_layer(plug_name, target_layer):
    """
    Find the specific blend node that corresponds to a layer.
    This traverses the blend node hierarchy.
    """
    # Start with the outermost blend node
    blend_nodes = cmds.listConnections(plug_name, source=True, destination=False,
                                     type='animBlendNodeAdditiveDL')
    if not blend_nodes:
        blend_nodes = cmds.listConnections(plug_name, source=True, destination=False,
                                        type='animBlendNodeAdditiveRotation')
    if not blend_nodes:
        blend_nodes = cmds.listConnections(plug_name, source=True, destination=False,
                                        type='animBlendNodeAdditiveScale')

    if not blend_nodes:
        return None

    # Recursively search blend nodes
    def check_blend_node(blend_node):
        # Check if this blend node is for our target layer
        weightB_conn = cmds.listConnections(f"{blend_node}.weightB",
                                           source=True, destination=False)
        if weightB_conn and cmds.nodeType(weightB_conn[0]) == 'animLayer':
            if weightB_conn[0] == target_layer:
                return blend_node

        # Check inputA for nested blend nodes
        for input_name in ['inputA', 'inputAX', 'inputAY', 'inputAZ']:
            try:
                inputA_conn = cmds.listConnections(f"{blend_node}.{input_name}",
                                                  source=True, destination=False)
                if inputA_conn and cmds.nodeType(inputA_conn[0]) in [
                    'animBlendNodeAdditiveDL',
                    'animBlendNodeAdditiveRotation',
                    'animBlendNodeAdditiveScale'
                ]:
                    result = check_blend_node(inputA_conn[0])
                    if result:
                        return result
            except:
                pass

        return None

    return check_blend_node(blend_nodes[0])



# ============================================================================
# Profiling section
# ============================================================================

#####
import time
import functools
from functools import wraps
import cProfile
import pstats
import io
from typing import Callable, Dict, List
from collections import defaultdict

# Profile timing storage
profile_times = defaultdict(list)


def cprofile_function(func):
    """
    Decorator using Python's built-in cProfile for detailed profiling

    Usage:
        @cprofile_function
        def my_function():
            pass
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()

        result = func(*args, **kwargs)

        pr.disable()

        # Create string buffer to capture output
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats()

        # Print the profiling results
        print(f"\n=== CPROFILE RESULTS FOR {func.__name__} ===")
        print(s.getvalue())
        print("=== END CPROFILE ===\n")

        return result
    return wrapper

# ============================================================================
# Run when executed
# ============================================================================
@cprofile_function
def run():
    cloneTransform('source1', 'target', start_frame=1, end_frame=1000, layer="AnimLayer2", euler_filter=True)
    # cloneTransform(start_frame=1, end_frame=1000, euler_filter=False)

if __name__ == "__main__":
    run()

