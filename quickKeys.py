from __future__ import absolute_import, print_function
import sys

import maya.api.OpenMaya as OpenMaya
import maya.api.OpenMayaAnim as OpenMayaAnim
import maya.cmds as cmds
from six.moves import range

maya_useNewAPI = True


class PyQuickKeysCommand(OpenMaya.MPxCommand):
    """
    OPTIMIZED: Build keys (and animCurves) on multiple attributes with the provided values.
    Uses bulk operations for massive performance gains.

    Supports animation layers for proper layer-aware keyframing.

    Arguments: Selection List
        A list of attributes to set data on
    -frames (-f) Time Multiple Optional
        The frames to set the keys on. If not provided, set only for the current frame
    -values (-v) Float Multiple
        The values to set to the attributes. Because of how Maya's mel commands are
        built, this must be a single flat list of values.
        If N attributes are provided, then the first N values of this list are applied
        to the attributes on the first given frame. The next N values are applied on the
        second given frame, etc...
    -uniform (-u) Bool Optional Default:False
        If True, then only take a single value per attribute and set it repeatedly
        on all the given frames
    -layer (-l) String Optional
        The animation layer to set keys on. If not provided, uses BaseAnimation.
        The layer must already exist.
    """

    kPluginVersion = "2.5.1"  # Fixed muted layer handling
    kPluginCmdName = "quickKeys"

    kFramesFlag, kFramesFlagLong = "-f", "-frames"
    kUniformFlag, kUniformFlagLong = "-u", "-uniform"
    kValuesFlag, kValuesFlagLong = "-v", "-values"
    kLayerFlag, kLayerFlagLong = "-l", "-layer"

    dependencyFn = OpenMaya.MFnDependencyNode()
    animCurveFn = OpenMayaAnim.MFnAnimCurve()

    normalCtx = OpenMaya.MDGContext.kNormal

    def __init__(self):
        super(PyQuickKeysCommand, self).__init__()

        self.times = OpenMaya.MTimeArray()
        self.values = None
        self.attrs = None
        self.curveTypes = None
        self.uniform = False
        self.layer = "BaseAnimation"

    def doIt(self, args):
        """
        Call once the first time command is run
        """
        parser = OpenMaya.MArgDatabase(self.syntax(), args)
        self.selectionList = parser.getObjectList()

        self.parseFlagArguments(parser)
        self.redoIt()

    def redoIt(self):
        """
        Call every subsequent time the command is run
        """
        self.animCurveChange = OpenMayaAnim.MAnimCurveChange()
        self.dgModifier = OpenMaya.MDGModifier()

        # Pre-validate all plugs
        for i in range(self.selectionList.length()):
            plug = self.selectionList.getPlug(i)
            if not plug.isKeyable:
                continue
            if plug.isLocked:
                raise RuntimeError(
                    "Plug {0} is Locked and cannot be modified".format(plug)
                )

        # Process all plugs
        for i in range(self.selectionList.length()):
            plug = self.selectionList.getPlug(i)
            if not plug.isKeyable:
                continue

            try:
                # Get or create animCurve on the specified layer
                animCurve = self.getOrCreateAnimCurveOnLayer(plug, self.curveTypes[i])

                if animCurve is None or animCurve.isNull():
                    raise RuntimeError("Got null animCurve for {}".format(plug.name()))

                # Build keys on animCurve - OPTIMIZED VERSION
                self.addKeysBulk(animCurve, self.values[i])

            except Exception as e:
                import traceback
                sys.stderr.write("Error processing plug {}: {}\n".format(plug.name(), str(e)))
                sys.stderr.write(traceback.format_exc())
                raise

        self.normalCtx.makeCurrent()

    def undoIt(self):
        """
        Undoes the command operations
        """
        self.animCurveChange.undoIt()
        self.dgModifier.undoIt()

    def isUndoable(self):
        """
        Indicate the command is undoable
        """
        return True

    def getOrCreateAnimCurveOnLayer(self, plug, curveType):
        """
        DISPATCHER: Routes to appropriate curve creation logic based on layer.

        For BaseAnimation: Uses simple direct connection
        For other layers: Uses complex blend node traversal

        Args:
            plug (MPlug): The attribute plug to key
            curveType: The MFnAnimCurve curve type

        Returns:
            MObject: The anim curve object
        """
        if self.layer == "BaseAnimation":
            # BaseAnimation: Direct connection, no blend nodes
            if OpenMayaAnim.MAnimUtil.isAnimated(plug):
                animCurves = OpenMayaAnim.MAnimUtil.findAnimation(plug)
                if animCurves and len(animCurves) > 0:
                    animCurve = animCurves[0]
                else:
                    raise RuntimeError("MAnimUtil said animated but no curve found for {}".format(plug.name()))
            else:
                # Create new curve - modifier will be executed later
                animCurve = self.animCurveFn.create(plug, curveType, self.dgModifier)

            return animCurve
        else:
            # Other layers: Need to traverse blend node hierarchy
            return self.getOrCreateLayerCurve(plug, curveType)

    def getOrCreateLayerCurve(self, plug, curveType):
        """
        Find or create an anim curve on a specific animation layer.

        STRICT MODE: This function requires the plug to already be on the SPECIFIC layer.
        It will NOT add the plug to the layer - that must be done beforehand.
        It WILL create the anim curve on the layer if one doesn't exist yet.

        Args:
            plug (MPlug): The attribute plug
            curveType: The curve type

        Returns:
            MObject: The anim curve on the layer

        Raises:
            RuntimeError: If layer doesn't exist or plug is not on the specific layer
        """
        # VALIDATE: Layer must exist
        if not cmds.objExists(self.layer):
            raise RuntimeError(
                "Animation layer '{0}' does not exist. "
                "Create it first with: cmds.animLayer('{0}')".format(self.layer)
            )

        plugName = plug.name()
        attrName = plugName.split('.')[-1]
        nodeName = plugName.split('.')[0]

        # CHECK: Does this plug have blend infrastructure (i.e., is it on ANY layer)?
        blendNodes = cmds.listConnections(plugName, source=True, destination=False,
                                         type='animBlendNodeAdditiveDL')

        if not blendNodes:
            blendNodes = cmds.listConnections(plugName, source=True, destination=False,
                                            type='animBlendNodeAdditiveRotation')

        if not blendNodes:
            blendNodes = cmds.listConnections(plugName, source=True, destination=False,
                                            type='animBlendNodeAdditiveScale')

        if not blendNodes:
            # No blend node = plug is not on any layer
            raise RuntimeError(
                "Attribute '{0}' is not on any animation layer. "
                "Add it to layer '{1}' first:\n"
                "  cmds.select('{2}')\n"
                "  cmds.animLayer('{1}', edit=True, addSelectedObjects=True)".format(
                    plugName, self.layer, nodeName
                )
            )

        blendNodeName = blendNodes[0]

        # CRITICAL: Validate this blend node is for the SPECIFIC layer we want
        # The layer is connected to the weightB attribute of the blend node
        weightB_conn = cmds.listConnections(blendNodeName + '.weightB',
                                           source=True, destination=False)

        if weightB_conn:
            actualLayer = weightB_conn[0]
            # Check if it's actually a layer node
            if cmds.nodeType(actualLayer) == 'animLayer':
                if actualLayer != self.layer:
                    raise RuntimeError(
                        "Attribute '{0}' is on layer '{1}', not on requested layer '{2}'. "
                        "Add it to layer '{2}' first or key it on layer '{1}'.".format(
                            plugName, actualLayer, self.layer
                        )
                    )

        # Get blend node as MObject
        selList = OpenMaya.MSelectionList()
        selList.add(blendNodeName)
        blendNodeObj = selList.getDependNode(0)
        blendNodeFn = OpenMaya.MFnDependencyNode(blendNodeObj)

        # Determine which input attribute to look for based on the attribute we're keying
        targetInputName = self.getTargetInputName(attrName, blendNodeFn)

        if not targetInputName:
            raise RuntimeError("Could not determine input name for {} on blend node {}".format(
                attrName, blendNodeName))

        # Find the input plug by name
        try:
            inputAttr = blendNodeFn.attribute(targetInputName)
            inputPlug = blendNodeFn.findPlug(inputAttr, False)
        except:
            raise RuntimeError("Could not find input plug '{}' on blend node {}".format(
                targetInputName, blendNodeName))

        # Check if an anim curve is connected
        if inputPlug.isConnected:
            # Get the existing connected anim curve
            sourcePlugs = inputPlug.connectedTo(True, False)
            if sourcePlugs:
                sourcePlug = sourcePlugs[0]
                sourceNode = sourcePlug.node()

                # Verify it's an anim curve
                if "animcurve" in sourceNode.apiTypeStr.lower():
                    return sourceNode

        # No anim curve exists yet - need to create one
        # CRITICAL FIX: Don't read "static offset" from the blend node input!
        #
        # The blend node input may contain:
        # 1. Values from MUTED layers below (should be ignored)
        # 2. Values from animated curves on layers below (already handled by delta calc)
        # 3. Actual static offsets on THIS layer (rare, handled by delta calc)
        #
        # The delta calculation in quickKeysHelper.py already accounts for all layers
        # properly (skipping muted ones). If we read and "preserve" the blend input
        # value here, we contaminate our keys with values from muted layers.
        #
        # Solution: Don't create any initialization key. Let addKeysBulk() set all
        # keys with the correct delta-calculated values. The delta calculation has
        # already composed all non-muted layers below this one.

        try:
            # Create the anim curve and connect it to the blend node input
            animCurve = self.animCurveFn.create(inputPlug, curveType, self.dgModifier)

            # CRITICAL: Execute the modifier to actually create the curve and connection
            self.dgModifier.doIt()

            # Curve is now created and connected - ready for keys
            # No initialization key needed - addKeysBulk will immediately set correct values
            # The delta calculation already accounted for everything below this layer

            return animCurve

        except Exception as e:
            raise RuntimeError("Failed to create anim curve for {}: {}".format(
                plugName, str(e)))

    def getTargetInputName(self, attrName, blendNodeFn):
        """
        Determine which input attribute name to use on the blend node.

        For rotation blend nodes: rotateX -> inputBX, rotateY -> inputBY, rotateZ -> inputBZ
        For translate/scale blend nodes: translateX/scaleX -> inputB (they use separate blend nodes per axis)

        Args:
            attrName (str): Attribute name like "rotateX" or "translateX"
            blendNodeFn (MFnDependencyNode): Function set for the blend node

        Returns:
            str: Input attribute name, or None if cannot determine
        """
        # Check what kind of blend node this is
        blendNodeType = OpenMaya.MFnDependencyNode(blendNodeFn.object()).typeName

        # For rotation blend nodes, we need axis-specific inputs
        if blendNodeType == "animBlendNodeAdditiveRotation":
            if attrName == "rotateX":
                return "inputBX"
            elif attrName == "rotateY":
                return "inputBY"
            elif attrName == "rotateZ":
                return "inputBZ"

        # For translate/scale blend nodes, they're per-axis so just use inputB
        elif blendNodeType in ["animBlendNodeAdditiveDL", "animBlendNodeAdditiveScale"]:
            return "inputB"

        return None

    def parseFlagArguments(self, parser):
        """
        Parse flag arguments, and format the values for setting curves
        """
        if parser.isFlagSet(self.kFramesFlag):
            self.times = self.getTimeArgs(parser, self.kFramesFlag)
        else:
            self.times.append(OpenMayaAnim.MAnimControl.currentTime())

        if parser.isFlagSet(self.kValuesFlag):
            self.values = self.getValueArgs(
                parser=parser, flagArgument=self.kValuesFlag
            )

        if parser.isFlagSet(self.kUniformFlag):
            self.uniform = parser.flagArgumentBool(self.kUniformFlag, 0)

        if parser.isFlagSet(self.kLayerFlag):
            self.layer = parser.flagArgumentString(self.kLayerFlag, 0)

        self.checkArguments()

        # convert the types of the value arguments
        # and group them into per-attribute chunks
        newVals = []
        self.curveTypes = []
        ta = OpenMaya.MFnUnitAttribute()
        for i in range(self.selectionList.length()):
            plug = self.selectionList.getPlug(i)
            ta.setObject(plug.attribute())
            try:
                unitType = ta.unitType()
            except RuntimeError:
                # it's an untyped attribute
                unitType = None

            # Get every Nth value from self.values for this attribute
            vals = self.values[i :: self.selectionList.length()]

            # Convert from uiUnits to internal units for distance and angle
            if unitType == OpenMaya.MFnUnitAttribute.kDistance:
                if OpenMaya.MDistance.uiUnit() != OpenMaya.MDistance.internalUnit():
                    vals = [OpenMaya.MDistance.uiToInternal(v) for v in vals]
                self.curveTypes.append(OpenMayaAnim.MFnAnimCurve.kAnimCurveTL)

            elif unitType == OpenMaya.MFnUnitAttribute.kAngle:
                if OpenMaya.MAngle.uiUnit() != OpenMaya.MAngle.internalUnit():
                    vals = [OpenMaya.MAngle.uiToInternal(v) for v in vals]
                self.curveTypes.append(OpenMayaAnim.MFnAnimCurve.kAnimCurveTA)

            else:
                self.curveTypes.append(OpenMayaAnim.MFnAnimCurve.kAnimCurveTU)

            newVals.append(vals)
        self.values = newVals

    def getTimeArgs(self, parser, flagArgument):
        """Read the -frames flag argument"""
        times = OpenMaya.MTimeArray()
        for occur in range(parser.numberOfFlagUses(self.kFramesFlag)):
            frame = parser.getFlagArgumentList(self.kFramesFlag, occur).asDouble(0)
            time = OpenMaya.MTime(frame, OpenMaya.MTime.uiUnit())
            times.append(time)
        return times

    def getValueArgs(self, parser, flagArgument):
        """Read the -value flag argument"""
        values = []
        for occur in range(parser.numberOfFlagUses(flagArgument)):
            arglist = parser.getFlagArgumentList(flagArgument, occur)
            values.append(arglist.asDouble(0))
        return values

    def checkArguments(self):
        """
        Ensure frame and transform arguments are correct in length.
        """
        if not self.values:
            raise ValueError("Values must be provided")

        if self.uniform:
            if self.selectionList.length() != len(self.values):
                raise ValueError(
                    "More than one value per attribute provided "
                    "Unable to set given uniformly throughout the frames."
                )
        else:
            requiredValues = self.selectionList.length() * len(self.times)
            if len(self.values) != requiredValues:
                msg = "{0} attributes and {1} times provided. "
                msg += "Therefore {0} * {1} = {2} values are required. Got {3}"
                raise ValueError(
                    msg.format(
                        self.selectionList.length(),
                        len(self.times),
                        requiredValues,
                        len(self.values),
                    )
                )

    def addKeys(self, animCurve, values):
        """
        LEGACY METHOD - kept for backward compatibility and debugging.
        Add values to animCurves using individual addKey calls (SLOW).

        In production, addKeysBulk() is used instead (much faster).
        This method can be useful for debugging if bulk operations have issues.
        """
        self.animCurveFn.setObject(animCurve)

        # Ensures time and value length are in-sync
        if len(self.times) == 1 or self.uniform:
            uniform = values[0]
            values = OpenMaya.MDoubleArray([uniform for num in range(len(self.times))])

        # Set keys and values
        for time, value in zip(self.times, values):
            index = self.animCurveFn.find(time)
            if not index:
                self.animCurveFn.addKey(time, value, change=self.animCurveChange)
                index = self.animCurveFn.find(time)

            self.animCurveFn.setValue(index, value, change=self.animCurveChange)

    def addKeysBulk(self, animCurve, values):
        """
        OPTIMIZED: Add values to animCurves using bulk operations (FAST).

        This is ~10-100x faster than addKeys() for large key counts.
        Uses addKeys() (plural) for new keys and setValue() for existing keys.
        This is the primary method used in production.
        """
        self.animCurveFn.setObject(animCurve)

        # Ensures time and value length are in-sync
        if len(self.times) == 1 or self.uniform:
            uniform = values[0]
            values = [uniform for _ in range(len(self.times))]

        # Separate new keys from existing keys
        new_times = OpenMaya.MTimeArray()
        new_values = OpenMaya.MDoubleArray()
        existing_updates = []  # List of (index, value) tuples

        for time, value in zip(self.times, values):
            index = self.animCurveFn.find(time)
            if index is None:
                # Key doesn't exist - add to bulk insert list
                new_times.append(time)
                new_values.append(value)
            else:
                # Key exists - queue for update
                existing_updates.append((index, value))

        # BULK INSERT: Add all new keys at once (MUCH faster than loop)
        if len(new_times) > 0:
            try:
                # Use addKeys (plural) for bulk insertion
                self.animCurveFn.addKeys(
                    new_times,
                    new_values,
                    OpenMayaAnim.MFnAnimCurve.kTangentGlobal,
                    OpenMayaAnim.MFnAnimCurve.kTangentGlobal,
                    False,  # keepExistingKeys
                    change=self.animCurveChange
                )
            except Exception as e:
                # Fallback to individual addKey if bulk operation fails
                sys.stderr.write("Bulk addKeys failed, falling back: {}\n".format(str(e)))
                for time, value in zip(new_times, new_values):
                    self.animCurveFn.addKey(time, value, change=self.animCurveChange)

        # UPDATE EXISTING: Set values for keys that already existed
        for index, value in existing_updates:
            self.animCurveFn.setValue(index, value, change=self.animCurveChange)

    @classmethod
    def syntaxCreator(cls):
        """
        Defines the flags and arguments of the command
        """
        syntax = OpenMaya.MSyntax()

        syntax.addFlag(cls.kFramesFlag, cls.kFramesFlagLong, syntax.kLong)
        syntax.addFlag(cls.kValuesFlag, cls.kValuesFlagLong, syntax.kLong)
        syntax.addFlag(cls.kUniformFlag, cls.kUniformFlagLong, syntax.kBoolean)
        syntax.addFlag(cls.kLayerFlag, cls.kLayerFlagLong, syntax.kString)

        syntax.makeFlagMultiUse(cls.kFramesFlag)
        syntax.makeFlagMultiUse(cls.kValuesFlag)

        syntax.setObjectType(syntax.kSelectionList, 1)

        return syntax

    @classmethod
    def cmdCreator(cls):
        """
        Create an instance of the command
        """
        return cls()


def initializePlugin(plugin):
    """
    Initialize the script plug-in
    """
    pluginFn = OpenMaya.MFnPlugin(
        plugin,
        "Blur",
        PyQuickKeysCommand.kPluginVersion,
        "Any",
    )

    try:
        pluginFn.registerCommand(
            PyQuickKeysCommand.kPluginCmdName,
            PyQuickKeysCommand.cmdCreator,
            PyQuickKeysCommand.syntaxCreator,
        )
    except RuntimeError:
        sys.stderr.write(
            "Failed to register command: {}\n".format(PyQuickKeysCommand.kPluginCmdName)
        )
        raise


def uninitializePlugin(plugin):
    """
    Uninitialize the script plug-in
    """
    pluginFn = OpenMaya.MFnPlugin(plugin)
    try:
        pluginFn.deregisterCommand(PyQuickKeysCommand.kPluginCmdName)
    except RuntimeError:
        sys.stderr.write(
            "Failed to unregister command: {}\n".format(
                PyQuickKeysCommand.kPluginCmdName
            )
        )
        raise
