/**
 * Scaffolding file used to store all the setups needed to run 
 * tests automatically generated by EvoSuite
 * Tue Sep 26 21:25:16 GMT 2023
 */

package com.google.javascript.jscomp;

import org.evosuite.runtime.annotation.EvoSuiteClassExclude;
import org.junit.BeforeClass;
import org.junit.Before;
import org.junit.After;
import org.junit.AfterClass;
import org.evosuite.runtime.sandbox.Sandbox;
import org.evosuite.runtime.sandbox.Sandbox.SandboxMode;

@EvoSuiteClassExclude
public class DiagnosticGroups_ESTest_scaffolding {

  @org.junit.Rule 
  public org.evosuite.runtime.vnet.NonFunctionalRequirementRule nfr = new org.evosuite.runtime.vnet.NonFunctionalRequirementRule();

  private static final java.util.Properties defaultProperties = (java.util.Properties) java.lang.System.getProperties().clone(); 

  private org.evosuite.runtime.thread.ThreadStopper threadStopper =  new org.evosuite.runtime.thread.ThreadStopper (org.evosuite.runtime.thread.KillSwitchHandler.getInstance(), 3000);


  @BeforeClass 
  public static void initEvoSuiteFramework() { 
    org.evosuite.runtime.RuntimeSettings.className = "com.google.javascript.jscomp.DiagnosticGroups"; 
    org.evosuite.runtime.GuiSupport.initialize(); 
    org.evosuite.runtime.RuntimeSettings.maxNumberOfThreads = 100; 
    org.evosuite.runtime.RuntimeSettings.maxNumberOfIterationsPerLoop = 10000; 
    org.evosuite.runtime.RuntimeSettings.mockSystemIn = true; 
    org.evosuite.runtime.RuntimeSettings.sandboxMode = org.evosuite.runtime.sandbox.Sandbox.SandboxMode.RECOMMENDED; 
    org.evosuite.runtime.sandbox.Sandbox.initializeSecurityManagerForSUT(); 
    org.evosuite.runtime.classhandling.JDKClassResetter.init();
    setSystemProperties();
    initializeClasses();
    org.evosuite.runtime.Runtime.getInstance().resetRuntime(); 
  } 

  @AfterClass 
  public static void clearEvoSuiteFramework(){ 
    Sandbox.resetDefaultSecurityManager(); 
    java.lang.System.setProperties((java.util.Properties) defaultProperties.clone()); 
  } 

  @Before 
  public void initTestCase(){ 
    threadStopper.storeCurrentThreads();
    threadStopper.startRecordingTime();
    org.evosuite.runtime.jvm.ShutdownHookHandler.getInstance().initHandler(); 
    org.evosuite.runtime.sandbox.Sandbox.goingToExecuteSUTCode(); 
    setSystemProperties(); 
    org.evosuite.runtime.GuiSupport.setHeadless(); 
    org.evosuite.runtime.Runtime.getInstance().resetRuntime(); 
    org.evosuite.runtime.agent.InstrumentingAgent.activate(); 
  } 

  @After 
  public void doneWithTestCase(){ 
    threadStopper.killAndJoinClientThreads();
    org.evosuite.runtime.jvm.ShutdownHookHandler.getInstance().safeExecuteAddedHooks(); 
    org.evosuite.runtime.classhandling.JDKClassResetter.reset(); 
    resetClasses(); 
    org.evosuite.runtime.sandbox.Sandbox.doneWithExecutingSUTCode(); 
    org.evosuite.runtime.agent.InstrumentingAgent.deactivate(); 
    org.evosuite.runtime.GuiSupport.restoreHeadlessMode(); 
  } 

  public static void setSystemProperties() {
 
    java.lang.System.setProperties((java.util.Properties) defaultProperties.clone()); 
    java.lang.System.setProperty("user.dir", "/data/lhy/TEval-plus"); 
    java.lang.System.setProperty("java.io.tmpdir", "/tmp"); 
  }

  private static void initializeClasses() {
    org.evosuite.runtime.classhandling.ClassStateSupport.initializeClasses(DiagnosticGroups_ESTest_scaffolding.class.getClassLoader() ,
      "com.google.common.collect.Sets$CartesianSet",
      "com.google.javascript.jscomp.CompilerOptions$LanguageMode",
      "com.google.javascript.jscomp.mozilla.rhino.ErrorReporter",
      "com.google.javascript.rhino.jstype.PrototypeObjectType",
      "com.google.common.collect.Lists$RandomAccessPartition",
      "com.google.common.collect.Collections2",
      "com.google.javascript.jscomp.mozilla.rhino.ast.Scope",
      "com.google.common.collect.PeekingIterator",
      "com.google.javascript.jscomp.NodeTraversal$Callback",
      "com.google.javascript.jscomp.CheckSideEffects",
      "com.google.javascript.jscomp.mozilla.rhino.EvaluatorException",
      "com.google.javascript.jscomp.graph.Graph",
      "com.google.javascript.rhino.jstype.StaticScope",
      "com.google.javascript.jscomp.PassFactory",
      "com.google.common.collect.Sets$2",
      "com.google.javascript.jscomp.mozilla.rhino.NativeArray",
      "com.google.common.collect.Sets$3",
      "com.google.javascript.jscomp.JSModule",
      "com.google.javascript.rhino.jstype.ObjectType",
      "com.google.common.collect.Sets$1",
      "com.google.javascript.jscomp.mozilla.rhino.ScriptRuntime$MessageProvider",
      "com.google.javascript.jscomp.mozilla.rhino.Node",
      "com.google.javascript.jscomp.mozilla.rhino.Icode",
      "com.google.common.collect.ImmutableSet$ArrayImmutableSet",
      "com.google.common.collect.RegularImmutableMap",
      "com.google.javascript.jscomp.ControlFlowGraph",
      "com.google.javascript.jscomp.graph.GraphvizGraph",
      "com.google.javascript.jscomp.mozilla.rhino.ContextFactory$Listener",
      "com.google.javascript.jscomp.mozilla.rhino.debug.DebuggableScript",
      "com.google.javascript.jscomp.Tracer",
      "com.google.common.collect.AbstractMapEntry",
      "com.google.common.collect.Iterators$12",
      "com.google.common.collect.Iterators$11",
      "com.google.javascript.rhino.jstype.JSType$1",
      "com.google.javascript.jscomp.ClosureCodingConvention",
      "com.google.common.base.Predicate",
      "com.google.javascript.jscomp.CodingConvention",
      "com.google.javascript.jscomp.VariableReferenceCheck",
      "com.google.javascript.jscomp.WarningsGuard",
      "com.google.javascript.jscomp.SourceMap",
      "com.google.common.base.Joiner",
      "com.google.javascript.jscomp.CheckAccessControls",
      "com.google.common.collect.SingletonImmutableMap",
      "com.google.javascript.jscomp.CompilerOptions",
      "com.google.javascript.jscomp.mozilla.rhino.ContextFactory$GlobalSetter",
      "com.google.common.collect.Iterators$14",
      "com.google.common.collect.Iterators$13",
      "com.google.common.collect.RegularImmutableMap$LinkedEntry",
      "com.google.javascript.jscomp.mozilla.rhino.Kit",
      "com.google.common.collect.Lists$Partition",
      "com.google.common.collect.Lists",
      "com.google.common.collect.UnmodifiableListIterator",
      "com.google.javascript.jscomp.RhinoErrorReporter",
      "com.google.javascript.rhino.ErrorReporter",
      "com.google.javascript.jscomp.DefaultCodingConvention",
      "com.google.javascript.jscomp.CheckGlobalNames",
      "org.mozilla.classfile.ClassFileWriter$ClassFileFormatException",
      "com.google.common.base.CharMatcher",
      "com.google.javascript.jscomp.mozilla.rhino.NativeCall",
      "com.google.common.base.Joiner$MapJoiner",
      "com.google.javascript.jscomp.CheckRegExp",
      "com.google.javascript.jscomp.mozilla.rhino.EcmaError",
      "com.google.javascript.jscomp.CssRenamingMap",
      "com.google.common.base.CharMatcher$5",
      "com.google.common.base.CharMatcher$4",
      "com.google.javascript.jscomp.CheckGlobalThis",
      "com.google.common.base.CharMatcher$3",
      "com.google.common.base.CharMatcher$2",
      "com.google.common.collect.RegularImmutableMap$NonTerminalEntry",
      "com.google.common.base.CharMatcher$9",
      "com.google.common.base.CharMatcher$8",
      "com.google.common.base.CharMatcher$7",
      "com.google.common.base.CharMatcher$6",
      "com.google.javascript.rhino.EcmaError",
      "com.google.javascript.jscomp.mozilla.rhino.xml.XMLObject",
      "com.google.common.base.Preconditions",
      "com.google.javascript.jscomp.MessageFormatter",
      "com.google.common.base.CharMatcher$1",
      "com.google.javascript.jscomp.mozilla.rhino.Context$ClassShutterSetter",
      "com.google.javascript.jscomp.parsing.Config",
      "com.google.common.collect.EmptyImmutableList",
      "com.google.common.collect.ImmutableEntry",
      "com.google.javascript.jscomp.mozilla.rhino.Evaluator",
      "com.google.common.base.Joiner$1",
      "com.google.common.base.Joiner$2",
      "com.google.common.collect.ImmutableCollection",
      "com.google.javascript.jscomp.mozilla.rhino.ast.Jump",
      "com.google.javascript.jscomp.PerformanceTracker",
      "com.google.javascript.jscomp.ProcessDefines",
      "com.google.javascript.rhino.ScriptRuntime",
      "com.google.javascript.jscomp.Result",
      "com.google.javascript.jscomp.CompilerPass",
      "com.google.javascript.jscomp.Scope",
      "com.google.javascript.jscomp.mozilla.rhino.Function",
      "com.google.common.collect.Iterators$6",
      "com.google.common.collect.BiMap",
      "com.google.common.collect.Iterators$7",
      "com.google.common.collect.ImmutableSet",
      "com.google.javascript.jscomp.CodeChangeHandler",
      "com.google.common.collect.Lists$AbstractListWrapper",
      "com.google.javascript.jscomp.FunctionTypeBuilder",
      "com.google.javascript.jscomp.mozilla.rhino.Scriptable",
      "com.google.javascript.jscomp.FunctionInformationMap",
      "com.google.javascript.jscomp.mozilla.rhino.ScriptableObject$GetterSlot",
      "com.google.common.collect.Iterators$1",
      "com.google.javascript.jscomp.DisambiguateProperties$Warnings",
      "com.google.common.collect.Iterators$2",
      "com.google.common.collect.Iterators$3",
      "com.google.javascript.jscomp.mozilla.rhino.RhinoException",
      "com.google.javascript.jscomp.mozilla.rhino.ast.FunctionNode",
      "com.google.common.base.CharMatcher$LookupTable",
      "com.google.common.collect.RegularImmutableList$1",
      "com.google.common.collect.Lists$StringAsImmutableList",
      "com.google.common.collect.Lists$2",
      "com.google.javascript.jscomp.ChainableReverseAbstractInterpreter",
      "com.google.javascript.jscomp.mozilla.rhino.optimizer.Codegen",
      "com.google.javascript.jscomp.ProcessTweaks",
      "com.google.javascript.jscomp.JSSourceFile",
      "com.google.common.collect.Lists$1",
      "com.google.javascript.jscomp.Compiler$3",
      "com.google.common.base.Supplier",
      "com.google.javascript.jscomp.mozilla.rhino.ScriptRuntime$DefaultMessageProvider",
      "com.google.javascript.jscomp.mozilla.rhino.JavaScriptException",
      "com.google.common.collect.EmptyImmutableSet",
      "com.google.javascript.jscomp.graph.LinkedDirectedGraph",
      "com.google.javascript.jscomp.mozilla.rhino.jdk13.VMBridge_jdk13",
      "com.google.javascript.jscomp.TypeValidator",
      "com.google.common.collect.ImmutableList",
      "com.google.javascript.jscomp.mozilla.rhino.ConstProperties",
      "com.google.protobuf.GeneratedMessage",
      "com.google.javascript.jscomp.mozilla.rhino.NativeContinuation",
      "com.google.javascript.jscomp.mozilla.rhino.ScriptRuntime$1",
      "com.google.protobuf.AbstractMessage",
      "com.google.javascript.jscomp.PassFactory$1",
      "com.google.javascript.jscomp.SemanticReverseAbstractInterpreter",
      "com.google.protobuf.MessageLite",
      "com.google.javascript.jscomp.mozilla.rhino.ast.ScriptNode",
      "com.google.common.collect.Maps$EntryTransformer",
      "com.google.javascript.jscomp.deps.SortedDependencies$MissingProvideException",
      "com.google.javascript.jscomp.CustomPassExecutionTime",
      "com.google.javascript.jscomp.SyntacticScopeCreator",
      "com.google.common.collect.Hashing",
      "com.google.javascript.jscomp.mozilla.rhino.BaseFunction",
      "com.google.javascript.jscomp.mozilla.rhino.ImporterTopLevel",
      "com.google.javascript.jscomp.SourceAst",
      "com.google.javascript.rhino.EvaluatorException",
      "com.google.javascript.jscomp.mozilla.rhino.IdScriptableObject",
      "com.google.common.collect.RegularImmutableList",
      "com.google.javascript.jscomp.mozilla.rhino.Callable",
      "com.google.javascript.jscomp.mozilla.rhino.ScriptableObject$Slot",
      "com.google.javascript.jscomp.mozilla.rhino.MemberBox",
      "com.google.javascript.jscomp.CheckUnreachableCode",
      "com.google.javascript.jscomp.mozilla.rhino.NativeObject",
      "com.google.javascript.jscomp.SourceExcerptProvider",
      "com.google.javascript.jscomp.ReferenceCollectingCallback$Behavior",
      "com.google.common.collect.Lists$TransformingRandomAccessList",
      "com.google.javascript.rhino.RhinoException",
      "com.google.common.collect.RegularImmutableMap$KeySet",
      "com.google.javascript.jscomp.mozilla.rhino.WrappedException",
      "com.google.javascript.rhino.Node",
      "com.google.javascript.jscomp.mozilla.rhino.ScriptableObject",
      "com.google.javascript.jscomp.mozilla.rhino.NativeFunction",
      "com.google.javascript.jscomp.ComposeWarningsGuard",
      "com.google.javascript.rhino.jstype.FunctionPrototypeType",
      "com.google.javascript.jscomp.VariableMap",
      "com.google.javascript.jscomp.PhaseOptimizer$Loop",
      "com.google.javascript.jscomp.JsAst",
      "com.google.javascript.jscomp.RhinoErrorReporter$NewRhinoErrorReporter",
      "com.google.common.collect.RegularImmutableSet",
      "com.google.javascript.jscomp.AbstractCompiler$LifeCycleStage",
      "com.google.javascript.rhino.Context",
      "com.google.javascript.jscomp.mozilla.rhino.IdFunctionCall",
      "com.google.javascript.jscomp.mozilla.rhino.debug.DebuggableObject",
      "com.google.javascript.rhino.jstype.JSType",
      "com.google.common.collect.ImmutableAsList",
      "com.google.common.collect.Sets$SetView",
      "com.google.javascript.jscomp.PassConfig",
      "com.google.javascript.jscomp.PhaseOptimizer",
      "com.google.common.collect.SingletonImmutableSet",
      "com.google.javascript.jscomp.DiagnosticGroups",
      "com.google.javascript.jscomp.ScopeCreator",
      "com.google.javascript.jscomp.mozilla.rhino.FunctionObject",
      "com.google.common.collect.RegularImmutableMap$TerminalEntry",
      "com.google.javascript.jscomp.graph.AdjacencyGraph",
      "com.google.javascript.jscomp.deps.SortedDependencies$CircularDependencyException",
      "com.google.javascript.jscomp.Compiler$CodeBuilder",
      "com.google.common.collect.Lists$TransformingSequentialList",
      "com.google.javascript.jscomp.SourceFile",
      "com.google.common.collect.AbstractIterator",
      "com.google.common.base.CharMatcher$And",
      "com.google.javascript.jscomp.DiagnosticType",
      "com.google.common.collect.MapDifference",
      "com.google.javascript.jscomp.CompilerInput",
      "com.google.javascript.rhino.jstype.FunctionType",
      "com.google.javascript.jscomp.AbstractCompiler",
      "com.google.common.collect.UnmodifiableIterator",
      "com.google.javascript.jscomp.Compiler",
      "com.google.javascript.jscomp.DiagnosticGroup",
      "com.google.javascript.jscomp.NodeTraversal$ScopedCallback",
      "com.google.javascript.jscomp.TypedScopeCreator",
      "com.google.common.collect.ImmutableSet$TransformedImmutableSet",
      "com.google.javascript.jscomp.mozilla.rhino.ast.AstNode",
      "com.google.javascript.jscomp.CodeChangeHandler$RecentChange",
      "com.google.javascript.jscomp.SyntacticScopeCreator$RedeclarationHandler",
      "com.google.common.base.CharMatcher$12",
      "com.google.common.base.CharMatcher$11",
      "com.google.common.base.CharMatcher$10",
      "com.google.javascript.jscomp.PhaseOptimizer$LoopInternal",
      "com.google.javascript.jscomp.JSError",
      "com.google.common.base.CharMatcher$15",
      "com.google.common.base.CharMatcher$14",
      "com.google.common.collect.ImmutableEnumSet",
      "com.google.common.base.CharMatcher$13",
      "com.google.common.collect.Lists$RandomAccessListWrapper",
      "com.google.common.base.Platform",
      "com.google.javascript.jscomp.TypedScopeCreator$GlobalScopeBuilder",
      "com.google.javascript.jscomp.mozilla.rhino.Context",
      "com.google.common.collect.ImmutableList$ReverseImmutableList",
      "com.google.protobuf.AbstractMessageLite",
      "com.google.javascript.jscomp.ErrorManager",
      "com.google.common.collect.SingletonImmutableList",
      "com.google.javascript.jscomp.CheckLevel",
      "com.google.common.base.Function",
      "com.google.common.collect.ImmutableMap",
      "com.google.common.collect.AbstractIndexedListIterator",
      "com.google.javascript.jscomp.JSModuleGraph",
      "com.google.javascript.jscomp.mozilla.rhino.Interpreter",
      "com.google.common.collect.Sets",
      "com.google.common.collect.ImmutableCollection$EmptyImmutableCollection",
      "com.google.javascript.jscomp.TypedScopeCreator$LocalScopeBuilder",
      "com.google.javascript.jscomp.Region",
      "com.google.javascript.jscomp.mozilla.rhino.ContextAction",
      "com.google.javascript.jscomp.NodeTraversal$AbstractPostOrderCallback",
      "com.google.javascript.jscomp.DefaultPassConfig",
      "com.google.javascript.jscomp.Compiler$IntermediateState",
      "com.google.javascript.jscomp.mozilla.rhino.InterpretedFunction",
      "com.google.javascript.jscomp.RhinoErrorReporter$OldRhinoErrorReporter",
      "com.google.javascript.jscomp.mozilla.rhino.ScriptRuntime",
      "com.google.common.collect.EmptyImmutableMap",
      "com.google.common.collect.Multimap",
      "com.google.common.collect.Iterators",
      "com.google.javascript.jscomp.VarCheck",
      "com.google.javascript.jscomp.mozilla.rhino.VMBridge",
      "com.google.javascript.jscomp.TypedScopeCreator$AbstractScopeBuilder",
      "com.google.javascript.jscomp.ProcessTweaks$TweakFunction",
      "com.google.javascript.jscomp.JSModuleGraph$ModuleDependenceException",
      "com.google.javascript.rhino.jstype.JSTypeRegistry",
      "com.google.javascript.jscomp.graph.DiGraph",
      "com.google.common.base.Platform$1",
      "com.google.javascript.jscomp.ReverseAbstractInterpreter",
      "com.google.common.collect.RegularImmutableMap$EntrySet",
      "com.google.javascript.jscomp.TypeCheck",
      "com.google.common.collect.Maps",
      "com.google.common.primitives.Ints",
      "com.google.javascript.jscomp.deps.DependencyInfo",
      "com.google.common.base.CharMatcher$Or",
      "com.google.javascript.jscomp.mozilla.rhino.Script",
      "com.google.protobuf.Message",
      "com.google.javascript.jscomp.mozilla.rhino.ContextFactory",
      "com.google.common.collect.RegularImmutableMap$Values",
      "com.google.javascript.jscomp.mozilla.rhino.jdk15.VMBridge_jdk15"
    );
  } 

  private static void resetClasses() {
    org.evosuite.runtime.classhandling.ClassResetter.getInstance().setClassLoader(DiagnosticGroups_ESTest_scaffolding.class.getClassLoader()); 

    org.evosuite.runtime.classhandling.ClassStateSupport.resetClasses(
      "com.google.common.base.Joiner",
      "com.google.common.base.Preconditions",
      "com.google.common.collect.Collections2",
      "com.google.common.base.Joiner$MapJoiner",
      "com.google.common.collect.Maps",
      "com.google.javascript.jscomp.DiagnosticType",
      "com.google.javascript.jscomp.CheckLevel",
      "com.google.javascript.jscomp.CheckGlobalThis",
      "com.google.javascript.jscomp.DiagnosticGroup",
      "com.google.common.collect.ImmutableCollection$EmptyImmutableCollection",
      "com.google.common.collect.ImmutableCollection",
      "com.google.common.collect.ImmutableSet",
      "com.google.common.collect.SingletonImmutableSet",
      "com.google.javascript.jscomp.CheckAccessControls",
      "com.google.common.collect.Hashing",
      "com.google.common.collect.ImmutableSet$ArrayImmutableSet",
      "com.google.common.collect.RegularImmutableSet",
      "com.google.javascript.jscomp.RhinoErrorReporter",
      "com.google.common.collect.Sets",
      "com.google.common.collect.UnmodifiableIterator",
      "com.google.common.collect.Iterators$1",
      "com.google.common.collect.Iterators$2",
      "com.google.common.collect.Iterators",
      "com.google.common.collect.UnmodifiableListIterator",
      "com.google.common.collect.AbstractIndexedListIterator",
      "com.google.common.collect.Iterators$11",
      "com.google.javascript.jscomp.TypeValidator",
      "com.google.javascript.jscomp.NodeTraversal$AbstractPostOrderCallback",
      "com.google.javascript.jscomp.VarCheck",
      "com.google.javascript.jscomp.CheckGlobalNames",
      "com.google.javascript.jscomp.VariableReferenceCheck",
      "com.google.common.primitives.Ints",
      "com.google.javascript.jscomp.ProcessDefines",
      "com.google.common.base.CharMatcher$11",
      "com.google.common.base.CharMatcher$12",
      "com.google.common.base.CharMatcher$Or",
      "com.google.common.base.Platform$1",
      "com.google.common.base.Platform",
      "com.google.common.base.CharMatcher$LookupTable",
      "com.google.common.base.CharMatcher$15",
      "com.google.common.base.CharMatcher$8",
      "com.google.common.base.CharMatcher$1",
      "com.google.common.base.CharMatcher$2",
      "com.google.common.base.CharMatcher$3",
      "com.google.common.base.CharMatcher$4",
      "com.google.common.base.CharMatcher$5",
      "com.google.common.base.CharMatcher$6",
      "com.google.common.base.CharMatcher$7",
      "com.google.common.base.CharMatcher",
      "com.google.javascript.jscomp.ProcessTweaks$TweakFunction",
      "com.google.javascript.jscomp.ProcessTweaks",
      "com.google.javascript.rhino.jstype.JSType$1",
      "com.google.javascript.rhino.jstype.JSType",
      "com.google.javascript.rhino.jstype.ObjectType",
      "com.google.javascript.jscomp.TypedScopeCreator",
      "com.google.javascript.jscomp.FunctionTypeBuilder",
      "com.google.javascript.jscomp.TypeCheck",
      "com.google.javascript.jscomp.CheckRegExp",
      "com.google.javascript.jscomp.SyntacticScopeCreator",
      "com.google.javascript.jscomp.CheckSideEffects",
      "com.google.javascript.jscomp.CheckUnreachableCode",
      "com.google.javascript.jscomp.DisambiguateProperties$Warnings",
      "com.google.javascript.jscomp.DiagnosticGroups",
      "com.google.common.collect.ImmutableMap",
      "com.google.common.collect.EmptyImmutableMap",
      "com.google.common.collect.AbstractMapEntry",
      "com.google.common.collect.ImmutableEntry",
      "com.google.common.collect.RegularImmutableMap",
      "com.google.common.collect.RegularImmutableMap$TerminalEntry",
      "com.google.common.collect.RegularImmutableMap$NonTerminalEntry",
      "com.google.javascript.jscomp.JSError",
      "com.google.javascript.jscomp.CollapseProperties",
      "com.google.javascript.jscomp.ObjectPropertyStringPreprocess",
      "com.google.common.collect.Iterators$13",
      "com.google.javascript.jscomp.SyntacticScopeCreator$DefaultRedeclarationHandler",
      "com.google.javascript.jscomp.NodeTraversal",
      "com.google.javascript.jscomp.JsMessageVisitor",
      "com.google.javascript.jscomp.ReplaceCssNames",
      "com.google.javascript.jscomp.ExternExportsPass",
      "com.google.javascript.jscomp.ProcessClosurePrimitives",
      "com.google.javascript.jscomp.ReplaceStrings",
      "com.google.javascript.jscomp.CheckPropertyOrder",
      "com.google.javascript.jscomp.AbstractPeepholeOptimization",
      "com.google.javascript.jscomp.PeepholeFoldConstants",
      "com.google.javascript.jscomp.ScopedAliases",
      "com.google.javascript.jscomp.DefaultCodingConvention",
      "com.google.javascript.jscomp.ClosureCodingConvention",
      "com.google.javascript.jscomp.GoogleCodingConvention",
      "com.google.javascript.jscomp.PeepholeFoldWithTypes",
      "com.google.javascript.rhino.Node",
      "com.google.javascript.rhino.Node$StringNode",
      "com.google.javascript.jscomp.AbstractMessageFormatter",
      "com.google.javascript.jscomp.LightweightMessageFormatter$LineNumberingFormatter",
      "com.google.javascript.jscomp.LightweightMessageFormatter",
      "com.google.javascript.jscomp.BasicErrorManager",
      "com.google.javascript.jscomp.PrintStreamErrorManager",
      "com.google.javascript.jscomp.BasicErrorManager$LeveledJSErrorComparator",
      "com.google.javascript.jscomp.AbstractCompiler",
      "com.google.javascript.jscomp.Compiler",
      "com.google.javascript.jscomp.AbstractCompiler$LifeCycleStage",
      "com.google.javascript.jscomp.RhinoErrorReporter$OldRhinoErrorReporter",
      "com.google.javascript.rhino.ScriptRuntime",
      "com.google.javascript.rhino.Context",
      "com.google.javascript.jscomp.mozilla.rhino.Kit",
      "com.google.javascript.jscomp.mozilla.rhino.optimizer.Codegen",
      "com.google.javascript.jscomp.mozilla.rhino.Icode",
      "com.google.javascript.jscomp.mozilla.rhino.Interpreter",
      "com.google.javascript.jscomp.mozilla.rhino.Context",
      "com.google.javascript.jscomp.mozilla.rhino.ContextFactory",
      "com.google.javascript.jscomp.mozilla.rhino.ScriptableObject$Slot",
      "com.google.javascript.jscomp.mozilla.rhino.ScriptableObject",
      "com.google.javascript.jscomp.mozilla.rhino.ScriptRuntime$DefaultMessageProvider",
      "com.google.javascript.jscomp.mozilla.rhino.ScriptRuntime",
      "com.google.javascript.jscomp.mozilla.rhino.jdk13.VMBridge_jdk13",
      "com.google.javascript.jscomp.mozilla.rhino.jdk15.VMBridge_jdk15",
      "com.google.javascript.jscomp.mozilla.rhino.VMBridge",
      "com.google.javascript.jscomp.RhinoErrorReporter$NewRhinoErrorReporter",
      "com.google.javascript.jscomp.PassFactory",
      "com.google.javascript.jscomp.Compiler$3",
      "com.google.javascript.jscomp.CodeChangeHandler$RecentChange",
      "com.google.common.collect.Lists",
      "com.google.javascript.jscomp.CompilerOptions$NullAliasTransformationHandler$NullAliasTransformation",
      "com.google.javascript.jscomp.CompilerOptions$NullAliasTransformationHandler",
      "com.google.javascript.jscomp.CompilerOptions",
      "com.google.common.collect.ImmutableList",
      "com.google.common.collect.EmptyImmutableList$1",
      "com.google.common.collect.EmptyImmutableList",
      "com.google.javascript.jscomp.SourceMap$DetailLevel",
      "com.google.javascript.jscomp.SourceMap$Format",
      "com.google.javascript.jscomp.CompilerOptions$LanguageMode",
      "com.google.javascript.jscomp.CompilerOptions$DevMode",
      "com.google.javascript.jscomp.VariableRenamingPolicy",
      "com.google.javascript.jscomp.PropertyRenamingPolicy",
      "com.google.javascript.jscomp.AnonymousFunctionNamingPolicy",
      "com.google.javascript.jscomp.CompilerOptions$TweakProcessing",
      "com.google.javascript.jscomp.CompilerOptions$TracerMode",
      "com.google.javascript.jscomp.ErrorFormat",
      "com.google.common.collect.AbstractMultimap",
      "com.google.common.collect.AbstractSetMultimap",
      "com.google.common.collect.AbstractSortedSetMultimap",
      "com.google.common.collect.TreeMultimap",
      "com.google.common.collect.Ordering",
      "com.google.common.collect.NaturalOrdering",
      "com.google.common.collect.AbstractMultimap$KeySet",
      "com.google.common.collect.AbstractMultimap$SortedKeySet",
      "com.google.javascript.jscomp.CreateSyntheticBlocks",
      "com.google.common.collect.EmptyImmutableSet",
      "com.google.javascript.jscomp.Compiler$CodeBuilder",
      "com.google.javascript.jscomp.StrictModeCheck",
      "com.google.javascript.jscomp.JSModule",
      "com.google.javascript.jscomp.SanityCheck",
      "com.google.javascript.rhino.SimpleErrorReporter",
      "com.google.javascript.jscomp.PureFunctionIdentifier",
      "com.google.javascript.jscomp.CheckRequiresForConstructors",
      "com.google.javascript.rhino.Node$PropListItem",
      "com.google.javascript.jscomp.ConstCheck",
      "com.google.javascript.jscomp.WarningsGuard",
      "com.google.javascript.jscomp.StrictWarningsGuard",
      "com.google.javascript.jscomp.LoggerErrorManager",
      "com.google.common.collect.RegularImmutableList",
      "com.google.javascript.jscomp.PhaseOptimizer",
      "com.google.common.collect.Iterators$12"
    );
  }
}
