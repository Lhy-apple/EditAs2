/**
 * Scaffolding file used to store all the setups needed to run 
 * tests automatically generated by EvoSuite
 * Tue Sep 26 14:40:55 GMT 2023
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
      "com.google.common.collect.Hashing",
      "com.google.common.collect.Sets$CartesianSet",
      "com.google.javascript.jscomp.mozilla.rhino.ErrorReporter",
      "com.google.javascript.rhino.jstype.PrototypeObjectType",
      "com.google.common.collect.Collections2",
      "com.google.common.collect.PeekingIterator",
      "com.google.javascript.jscomp.NodeTraversal$Callback",
      "com.google.javascript.jscomp.CheckSideEffects",
      "com.google.common.collect.RegularImmutableList",
      "com.google.javascript.jscomp.graph.Graph",
      "com.google.javascript.rhino.jstype.StaticScope",
      "com.google.javascript.jscomp.CheckUnreachableCode",
      "com.google.javascript.jscomp.SourceExcerptProvider",
      "com.google.javascript.jscomp.ReferenceCollectingCallback$Behavior",
      "com.google.common.collect.Sets$2",
      "com.google.common.collect.RegularImmutableMap$KeySet",
      "com.google.common.collect.Sets$3",
      "com.google.javascript.rhino.jstype.ObjectType",
      "com.google.common.collect.Sets$1",
      "com.google.common.collect.ImmutableSet$ArrayImmutableSet",
      "com.google.common.collect.RegularImmutableMap",
      "com.google.javascript.jscomp.ControlFlowGraph",
      "com.google.javascript.jscomp.graph.GraphvizGraph",
      "com.google.javascript.rhino.jstype.FunctionPrototypeType",
      "com.google.common.collect.RegularImmutableSet",
      "com.google.common.collect.AbstractMapEntry",
      "com.google.common.collect.Iterators$12",
      "com.google.common.collect.Iterators$11",
      "com.google.javascript.rhino.jstype.JSType$1",
      "com.google.common.base.Predicate",
      "com.google.javascript.jscomp.VariableReferenceCheck",
      "com.google.common.base.Joiner",
      "com.google.javascript.jscomp.CheckAccessControls",
      "com.google.common.collect.SingletonImmutableMap",
      "com.google.javascript.rhino.jstype.JSType",
      "com.google.javascript.jscomp.CompilerOptions",
      "com.google.common.collect.ImmutableAsList",
      "com.google.common.collect.Sets$SetView",
      "com.google.common.collect.SingletonImmutableSet",
      "com.google.common.collect.Iterators$14",
      "com.google.common.collect.Iterators$13",
      "com.google.javascript.jscomp.DiagnosticGroups",
      "com.google.common.collect.RegularImmutableMap$LinkedEntry",
      "com.google.javascript.jscomp.ScopeCreator",
      "com.google.common.collect.UnmodifiableListIterator",
      "com.google.common.collect.RegularImmutableMap$TerminalEntry",
      "com.google.javascript.jscomp.graph.AdjacencyGraph",
      "com.google.javascript.jscomp.RhinoErrorReporter",
      "com.google.javascript.rhino.ErrorReporter",
      "com.google.javascript.jscomp.CheckGlobalNames",
      "com.google.common.collect.AbstractIterator",
      "com.google.common.base.CharMatcher",
      "com.google.common.base.Joiner$MapJoiner",
      "com.google.common.base.CharMatcher$And",
      "com.google.javascript.jscomp.CheckRegExp",
      "com.google.javascript.jscomp.DiagnosticType",
      "com.google.common.base.CharMatcher$5",
      "com.google.common.collect.MapDifference",
      "com.google.common.base.CharMatcher$4",
      "com.google.javascript.jscomp.CheckGlobalThis",
      "com.google.common.base.CharMatcher$3",
      "com.google.common.base.CharMatcher$2",
      "com.google.common.collect.RegularImmutableMap$NonTerminalEntry",
      "com.google.common.base.CharMatcher$9",
      "com.google.common.base.CharMatcher$8",
      "com.google.common.base.CharMatcher$7",
      "com.google.common.base.CharMatcher$6",
      "com.google.javascript.rhino.jstype.FunctionType",
      "com.google.javascript.jscomp.AbstractCompiler",
      "com.google.common.base.Preconditions",
      "com.google.common.collect.UnmodifiableIterator",
      "com.google.common.base.CharMatcher$1",
      "com.google.javascript.jscomp.Compiler",
      "com.google.common.collect.ImmutableEntry",
      "com.google.common.base.Joiner$1",
      "com.google.common.base.Joiner$2",
      "com.google.javascript.jscomp.DiagnosticGroup",
      "com.google.javascript.jscomp.NodeTraversal$ScopedCallback",
      "com.google.javascript.jscomp.TypedScopeCreator",
      "com.google.common.collect.ImmutableSet$TransformedImmutableSet",
      "com.google.javascript.jscomp.SyntacticScopeCreator$RedeclarationHandler",
      "com.google.common.base.CharMatcher$12",
      "com.google.common.collect.ImmutableCollection",
      "com.google.common.base.CharMatcher$11",
      "com.google.common.base.CharMatcher$10",
      "com.google.javascript.jscomp.JSError",
      "com.google.common.base.CharMatcher$15",
      "com.google.common.base.CharMatcher$14",
      "com.google.common.collect.ImmutableEnumSet",
      "com.google.javascript.jscomp.ProcessDefines",
      "com.google.common.base.CharMatcher$13",
      "com.google.common.base.Platform",
      "com.google.javascript.jscomp.TypedScopeCreator$GlobalScopeBuilder",
      "com.google.javascript.jscomp.CompilerPass",
      "com.google.javascript.jscomp.CheckLevel",
      "com.google.common.collect.Iterators$6",
      "com.google.common.collect.BiMap",
      "com.google.common.collect.Iterators$7",
      "com.google.common.base.Function",
      "com.google.common.collect.ImmutableSet",
      "com.google.javascript.jscomp.FunctionTypeBuilder",
      "com.google.common.collect.ImmutableMap",
      "com.google.common.collect.AbstractIndexedListIterator",
      "com.google.common.collect.Iterators$1",
      "com.google.javascript.jscomp.DisambiguateProperties$Warnings",
      "com.google.common.collect.Iterators$2",
      "com.google.common.collect.Iterators$3",
      "com.google.common.collect.Sets",
      "com.google.common.collect.ImmutableCollection$EmptyImmutableCollection",
      "com.google.javascript.jscomp.TypedScopeCreator$LocalScopeBuilder",
      "com.google.common.base.CharMatcher$LookupTable",
      "com.google.javascript.jscomp.ProcessTweaks",
      "com.google.javascript.jscomp.NodeTraversal$AbstractPostOrderCallback",
      "com.google.common.collect.EmptyImmutableMap",
      "com.google.common.collect.Multimap",
      "com.google.common.collect.EmptyImmutableSet",
      "com.google.common.collect.Iterators",
      "com.google.javascript.jscomp.graph.LinkedDirectedGraph",
      "com.google.javascript.jscomp.VarCheck",
      "com.google.javascript.jscomp.TypeValidator",
      "com.google.javascript.jscomp.TypedScopeCreator$AbstractScopeBuilder",
      "com.google.common.collect.ImmutableList",
      "com.google.javascript.jscomp.ProcessTweaks$TweakFunction",
      "com.google.javascript.jscomp.graph.DiGraph",
      "com.google.common.base.Platform$1",
      "com.google.common.collect.RegularImmutableMap$EntrySet",
      "com.google.javascript.jscomp.TypeCheck",
      "com.google.common.collect.Maps",
      "com.google.common.primitives.Ints",
      "com.google.common.base.CharMatcher$Or",
      "com.google.common.collect.Maps$EntryTransformer",
      "com.google.javascript.jscomp.SyntacticScopeCreator",
      "com.google.common.collect.RegularImmutableMap$Values"
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
      "com.google.javascript.jscomp.AbstractCompiler",
      "com.google.javascript.jscomp.Compiler",
      "com.google.javascript.jscomp.AbstractCompiler$LifeCycleStage",
      "com.google.javascript.jscomp.DefaultCodingConvention",
      "com.google.javascript.jscomp.ClosureCodingConvention",
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
      "com.google.common.collect.RegularImmutableMap",
      "com.google.common.collect.AbstractMapEntry",
      "com.google.common.collect.ImmutableEntry",
      "com.google.common.collect.RegularImmutableMap$TerminalEntry",
      "com.google.common.collect.RegularImmutableMap$NonTerminalEntry",
      "com.google.javascript.jscomp.RhinoErrorReporter$NewRhinoErrorReporter",
      "com.google.javascript.jscomp.PassFactory",
      "com.google.javascript.jscomp.Compiler$3",
      "com.google.javascript.jscomp.CodeChangeHandler$RecentChange",
      "com.google.common.collect.Lists",
      "com.google.javascript.jscomp.MoveFunctionDeclarations",
      "com.google.javascript.jscomp.NodeTraversal",
      "com.google.javascript.jscomp.SyntacticScopeCreator$DefaultRedeclarationHandler",
      "com.google.common.collect.Iterators$13",
      "com.google.javascript.jscomp.CompilerOptions$NullAliasTransformationHandler$NullAliasTransformation",
      "com.google.javascript.jscomp.CompilerOptions$NullAliasTransformationHandler",
      "com.google.javascript.jscomp.CompilerOptions",
      "com.google.common.collect.ImmutableList",
      "com.google.common.collect.EmptyImmutableList$1",
      "com.google.common.collect.EmptyImmutableList",
      "com.google.javascript.jscomp.SourceMap$Format",
      "com.google.javascript.jscomp.CompilerOptions$LanguageMode",
      "com.google.javascript.jscomp.CompilerOptions$DevMode",
      "com.google.javascript.jscomp.VariableRenamingPolicy",
      "com.google.javascript.jscomp.PropertyRenamingPolicy",
      "com.google.javascript.jscomp.AnonymousFunctionNamingPolicy",
      "com.google.javascript.jscomp.CompilerOptions$TweakProcessing",
      "com.google.javascript.jscomp.CompilerOptions$TracerMode",
      "com.google.javascript.jscomp.ErrorFormat",
      "com.google.common.collect.RegularImmutableList",
      "com.google.javascript.jscomp.PhaseOptimizer",
      "com.google.common.collect.Iterators$12",
      "com.google.javascript.jscomp.JsMessageVisitor",
      "com.google.javascript.jscomp.ProcessClosurePrimitives",
      "com.google.javascript.jscomp.RenameProperties$1",
      "com.google.javascript.jscomp.RenameProperties",
      "com.google.javascript.jscomp.StrictModeCheck",
      "com.google.javascript.jscomp.BasicErrorManager",
      "com.google.javascript.jscomp.PrintStreamErrorManager",
      "com.google.javascript.jscomp.AbstractMessageFormatter",
      "com.google.javascript.jscomp.LightweightMessageFormatter$LineNumberingFormatter",
      "com.google.javascript.jscomp.LightweightMessageFormatter",
      "com.google.javascript.jscomp.SourceExcerptProvider$SourceExcerpt",
      "com.google.javascript.jscomp.BasicErrorManager$LeveledJSErrorComparator",
      "com.google.common.collect.EmptyImmutableSet",
      "com.google.javascript.jscomp.ScopedAliases",
      "com.google.javascript.jscomp.ReferenceCollectingCallback$1",
      "com.google.javascript.jscomp.ReferenceCollectingCallback",
      "com.google.common.base.Predicates",
      "com.google.common.base.Predicates$ObjectPredicate",
      "com.google.javascript.jscomp.PassConfig",
      "com.google.javascript.jscomp.DefaultPassConfig",
      "com.google.javascript.jscomp.CrossModuleMethodMotion$IdGenerator",
      "com.google.javascript.jscomp.DefaultPassConfig$1",
      "com.google.javascript.jscomp.DefaultPassConfig$2",
      "com.google.javascript.jscomp.DefaultPassConfig$3",
      "com.google.javascript.jscomp.DefaultPassConfig$4",
      "com.google.javascript.jscomp.DefaultPassConfig$5",
      "com.google.javascript.jscomp.DefaultPassConfig$6",
      "com.google.javascript.jscomp.DefaultPassConfig$7",
      "com.google.javascript.jscomp.DefaultPassConfig$8",
      "com.google.javascript.jscomp.DefaultPassConfig$9",
      "com.google.javascript.jscomp.DefaultPassConfig$10",
      "com.google.javascript.jscomp.DefaultPassConfig$11",
      "com.google.javascript.jscomp.DefaultPassConfig$12",
      "com.google.javascript.jscomp.DefaultPassConfig$13",
      "com.google.javascript.jscomp.DefaultPassConfig$14",
      "com.google.javascript.jscomp.DefaultPassConfig$15",
      "com.google.javascript.jscomp.DefaultPassConfig$16",
      "com.google.javascript.jscomp.DefaultPassConfig$17",
      "com.google.javascript.jscomp.DefaultPassConfig$18",
      "com.google.javascript.jscomp.DefaultPassConfig$19",
      "com.google.javascript.jscomp.DefaultPassConfig$20",
      "com.google.javascript.jscomp.DefaultPassConfig$21",
      "com.google.javascript.jscomp.DefaultPassConfig$22",
      "com.google.javascript.jscomp.DefaultPassConfig$23",
      "com.google.javascript.jscomp.DefaultPassConfig$24",
      "com.google.javascript.jscomp.DefaultPassConfig$25",
      "com.google.javascript.jscomp.DefaultPassConfig$26",
      "com.google.javascript.jscomp.DefaultPassConfig$27",
      "com.google.javascript.jscomp.DefaultPassConfig$28",
      "com.google.javascript.jscomp.DefaultPassConfig$29",
      "com.google.javascript.jscomp.DefaultPassConfig$30",
      "com.google.javascript.jscomp.DefaultPassConfig$31",
      "com.google.javascript.jscomp.DefaultPassConfig$32",
      "com.google.javascript.jscomp.DefaultPassConfig$33",
      "com.google.javascript.jscomp.DefaultPassConfig$34",
      "com.google.javascript.jscomp.DefaultPassConfig$35",
      "com.google.javascript.jscomp.DefaultPassConfig$36",
      "com.google.javascript.jscomp.DefaultPassConfig$37",
      "com.google.javascript.jscomp.DefaultPassConfig$38",
      "com.google.javascript.jscomp.DefaultPassConfig$39",
      "com.google.javascript.jscomp.DefaultPassConfig$40",
      "com.google.javascript.jscomp.DefaultPassConfig$41",
      "com.google.javascript.jscomp.DefaultPassConfig$42",
      "com.google.javascript.jscomp.DefaultPassConfig$43",
      "com.google.javascript.jscomp.DefaultPassConfig$44",
      "com.google.javascript.jscomp.DefaultPassConfig$45",
      "com.google.javascript.jscomp.DefaultPassConfig$46",
      "com.google.javascript.jscomp.DefaultPassConfig$47",
      "com.google.javascript.jscomp.DefaultPassConfig$48",
      "com.google.javascript.jscomp.DefaultPassConfig$49",
      "com.google.javascript.jscomp.DefaultPassConfig$50",
      "com.google.javascript.jscomp.DefaultPassConfig$51",
      "com.google.javascript.jscomp.DefaultPassConfig$52",
      "com.google.javascript.jscomp.DefaultPassConfig$53",
      "com.google.javascript.jscomp.DefaultPassConfig$54",
      "com.google.javascript.jscomp.DefaultPassConfig$55",
      "com.google.javascript.jscomp.DefaultPassConfig$56",
      "com.google.javascript.jscomp.DefaultPassConfig$57",
      "com.google.javascript.jscomp.DefaultPassConfig$58",
      "com.google.javascript.jscomp.DefaultPassConfig$59",
      "com.google.javascript.jscomp.DefaultPassConfig$60",
      "com.google.javascript.jscomp.DefaultPassConfig$61",
      "com.google.javascript.jscomp.DefaultPassConfig$62",
      "com.google.javascript.jscomp.DefaultPassConfig$63",
      "com.google.javascript.jscomp.DefaultPassConfig$64",
      "com.google.javascript.jscomp.DefaultPassConfig$65",
      "com.google.javascript.jscomp.DefaultPassConfig$66",
      "com.google.javascript.jscomp.DefaultPassConfig$67",
      "com.google.javascript.jscomp.DefaultPassConfig$68",
      "com.google.javascript.jscomp.DefaultPassConfig$69",
      "com.google.javascript.jscomp.DefaultPassConfig$70",
      "com.google.javascript.jscomp.DefaultPassConfig$71",
      "com.google.javascript.jscomp.DefaultPassConfig$72",
      "com.google.javascript.jscomp.DefaultPassConfig$73",
      "com.google.javascript.jscomp.DefaultPassConfig$74",
      "com.google.javascript.jscomp.DefaultPassConfig$75",
      "com.google.javascript.jscomp.DefaultPassConfig$76",
      "com.google.javascript.jscomp.DefaultPassConfig$77",
      "com.google.javascript.jscomp.DefaultPassConfig$78",
      "com.google.javascript.jscomp.DefaultPassConfig$79",
      "com.google.javascript.jscomp.DefaultPassConfig$80",
      "com.google.javascript.jscomp.DefaultPassConfig$81",
      "com.google.javascript.jscomp.DefaultPassConfig$82",
      "com.google.javascript.jscomp.DefaultPassConfig$83",
      "com.google.javascript.jscomp.DefaultPassConfig$84",
      "com.google.javascript.jscomp.DefaultPassConfig$85",
      "com.google.javascript.jscomp.DefaultPassConfig$86",
      "com.google.javascript.jscomp.DefaultPassConfig$90",
      "com.google.javascript.jscomp.DefaultPassConfig$91",
      "com.google.javascript.jscomp.ExternExportsPass",
      "com.google.javascript.jscomp.ReplaceIdGenerators",
      "com.google.javascript.jscomp.CheckMissingReturn$1",
      "com.google.javascript.jscomp.CheckMissingReturn$2",
      "com.google.javascript.jscomp.CheckMissingReturn",
      "com.google.javascript.jscomp.AbstractPeepholeOptimization",
      "com.google.javascript.jscomp.PeepholeFoldConstants",
      "com.google.javascript.jscomp.FunctionNames",
      "com.google.javascript.jscomp.FunctionNames$FunctionListExtractor",
      "com.google.javascript.jscomp.RecordFunctionInformation",
      "com.google.protobuf.AbstractMessageLite",
      "com.google.protobuf.AbstractMessage",
      "com.google.protobuf.GeneratedMessage",
      "com.google.protobuf.UnknownFieldSet",
      "com.google.javascript.jscomp.FunctionInfo$1",
      "com.google.protobuf.Descriptors$FileDescriptor",
      "com.google.protobuf.DescriptorProtos$1",
      "com.google.protobuf.AbstractMessageLite$Builder",
      "com.google.protobuf.AbstractMessage$Builder",
      "com.google.protobuf.GeneratedMessage$Builder",
      "com.google.protobuf.DescriptorProtos$FileDescriptorProto$Builder",
      "com.google.protobuf.GeneratedMessage$ExtendableMessage",
      "com.google.protobuf.FieldSet",
      "com.google.protobuf.DescriptorProtos$FileOptions$OptimizeMode$1",
      "com.google.protobuf.DescriptorProtos$FileOptions$OptimizeMode",
      "com.google.protobuf.DescriptorProtos$FileOptions",
      "com.google.protobuf.CodedInputStream",
      "com.google.protobuf.ExtensionRegistryLite",
      "com.google.protobuf.ExtensionRegistry",
      "com.google.protobuf.UnknownFieldSet$Builder",
      "com.google.protobuf.WireFormat",
      "com.google.protobuf.DescriptorProtos$MessageOptions",
      "com.google.protobuf.DescriptorProtos$DescriptorProto",
      "com.google.protobuf.DescriptorProtos$DescriptorProto$Builder",
      "com.google.protobuf.DescriptorProtos$FieldDescriptorProto$Label$1",
      "com.google.protobuf.DescriptorProtos$FieldDescriptorProto$Label",
      "com.google.protobuf.DescriptorProtos$FieldDescriptorProto$Type$1",
      "com.google.protobuf.DescriptorProtos$FieldDescriptorProto$Type",
      "com.google.protobuf.DescriptorProtos$FieldOptions$CType$1",
      "com.google.protobuf.DescriptorProtos$FieldOptions$CType",
      "com.google.protobuf.DescriptorProtos$FieldOptions",
      "com.google.protobuf.DescriptorProtos$FieldDescriptorProto",
      "com.google.protobuf.DescriptorProtos$FieldDescriptorProto$Builder",
      "com.google.protobuf.DescriptorProtos$EnumOptions",
      "com.google.protobuf.DescriptorProtos$EnumDescriptorProto",
      "com.google.protobuf.DescriptorProtos$EnumDescriptorProto$Builder",
      "com.google.protobuf.DescriptorProtos$EnumValueOptions",
      "com.google.protobuf.DescriptorProtos$EnumValueDescriptorProto",
      "com.google.protobuf.DescriptorProtos$EnumValueDescriptorProto$Builder",
      "com.google.protobuf.DescriptorProtos$DescriptorProto$ExtensionRange",
      "com.google.protobuf.DescriptorProtos$DescriptorProto$ExtensionRange$Builder",
      "com.google.protobuf.GeneratedMessage$ExtendableBuilder",
      "com.google.protobuf.DescriptorProtos$FileOptions$Builder",
      "com.google.protobuf.Descriptors$DescriptorPool",
      "com.google.protobuf.Descriptors$DescriptorPool$PackageDescriptor",
      "com.google.protobuf.Descriptors$Descriptor",
      "com.google.protobuf.Descriptors",
      "com.google.protobuf.ByteString",
      "com.google.protobuf.WireFormat$JavaType",
      "com.google.protobuf.WireFormat$FieldType",
      "com.google.protobuf.Descriptors$FieldDescriptor$JavaType",
      "com.google.protobuf.Descriptors$FieldDescriptor$Type",
      "com.google.protobuf.Descriptors$FieldDescriptor",
      "com.google.protobuf.Descriptors$EnumDescriptor",
      "com.google.protobuf.Descriptors$EnumValueDescriptor",
      "com.google.protobuf.Descriptors$DescriptorPool$DescriptorIntPair",
      "com.google.protobuf.Descriptors$1",
      "com.google.protobuf.GeneratedMessage$FieldAccessorTable",
      "com.google.protobuf.GeneratedMessage$FieldAccessorTable$RepeatedFieldAccessor",
      "com.google.protobuf.GeneratedMessage$FieldAccessorTable$RepeatedMessageFieldAccessor",
      "com.google.protobuf.GeneratedMessage$FieldAccessorTable$SingularFieldAccessor",
      "com.google.protobuf.GeneratedMessage$FieldAccessorTable$SingularMessageFieldAccessor",
      "com.google.protobuf.GeneratedMessage$FieldAccessorTable$SingularEnumFieldAccessor",
      "com.google.protobuf.DescriptorProtos",
      "com.google.protobuf.DescriptorProtos$FileDescriptorProto",
      "com.google.javascript.jscomp.FunctionInfo",
      "com.google.javascript.jscomp.FunctionInformationMap",
      "com.google.javascript.jscomp.FunctionInformationMap$Builder",
      "com.google.javascript.rhino.SimpleErrorReporter",
      "com.google.javascript.rhino.jstype.JSTypeRegistry",
      "com.google.common.collect.AbstractMultimap",
      "com.google.common.collect.AbstractSetMultimap",
      "com.google.common.collect.LinkedHashMultimap",
      "com.google.common.collect.AbstractListMultimap",
      "com.google.common.collect.ArrayListMultimap",
      "com.google.javascript.rhino.jstype.JSTypeRegistry$ResolveMode",
      "com.google.javascript.rhino.jstype.JSTypeNative",
      "com.google.javascript.rhino.jstype.ValueType",
      "com.google.javascript.rhino.jstype.BooleanType",
      "com.google.javascript.rhino.jstype.NullType",
      "com.google.javascript.rhino.jstype.NumberType",
      "com.google.javascript.rhino.jstype.StringType",
      "com.google.javascript.rhino.jstype.UnknownType",
      "com.google.javascript.rhino.jstype.VoidType",
      "com.google.javascript.rhino.jstype.AllType",
      "com.google.javascript.rhino.jstype.PrototypeObjectType",
      "com.google.javascript.rhino.jstype.FunctionPrototypeType",
      "com.google.javascript.rhino.jstype.FunctionType",
      "com.google.javascript.rhino.jstype.FunctionParamBuilder",
      "com.google.javascript.rhino.Node",
      "com.google.javascript.rhino.Node$StringNode",
      "com.google.javascript.rhino.Node$PropListItem",
      "com.google.javascript.rhino.jstype.ArrowType",
      "com.google.javascript.rhino.jstype.FunctionType$Kind",
      "com.google.javascript.rhino.jstype.InstanceObjectType",
      "com.google.javascript.rhino.jstype.UnionTypeBuilder$1",
      "com.google.javascript.rhino.jstype.UnionTypeBuilder",
      "com.google.javascript.rhino.jstype.NoObjectType",
      "com.google.javascript.rhino.jstype.NoType",
      "com.google.javascript.rhino.jstype.NoResolvedType",
      "com.google.javascript.rhino.jstype.ErrorFunctionType",
      "com.google.javascript.rhino.jstype.UnionType",
      "com.google.javascript.rhino.jstype.FunctionBuilder",
      "com.google.javascript.rhino.jstype.JSTypeRegistry$1",
      "com.google.javascript.jscomp.ReplaceCssNames",
      "com.google.javascript.jscomp.CheckRequiresForConstructors",
      "com.google.javascript.jscomp.SourceFile",
      "com.google.javascript.jscomp.JSSourceFile",
      "com.google.common.base.Charsets",
      "com.google.javascript.jscomp.SourceFile$OnDisk",
      "com.google.javascript.jscomp.JSModuleGraph",
      "com.google.javascript.jscomp.Normalize",
      "com.google.javascript.jscomp.ObjectPropertyStringPreprocess",
      "com.google.javascript.jscomp.LoggerErrorManager",
      "com.google.javascript.jscomp.WarningsGuard",
      "com.google.javascript.jscomp.DiagnosticGroupWarningsGuard",
      "com.google.javascript.jscomp.ComposeWarningsGuard",
      "com.google.javascript.jscomp.ComposeWarningsGuard$1",
      "com.google.javascript.jscomp.WarningsGuard$Priority",
      "com.google.javascript.jscomp.SuppressDocWarningsGuard",
      "com.google.common.collect.RegularImmutableMap$EntrySet",
      "com.google.javascript.jscomp.CompilerInput",
      "com.google.javascript.jscomp.JsAst",
      "com.google.common.collect.ImmutableMultimap",
      "com.google.common.collect.ImmutableListMultimap",
      "com.google.common.collect.ImmutableMultimap$Builder",
      "com.google.common.collect.ImmutableListMultimap$Builder",
      "com.google.common.collect.ImmutableMultimap$BuilderMultimap",
      "com.google.javascript.jscomp.PureFunctionIdentifier",
      "com.google.javascript.jscomp.JSError"
    );
  }
}