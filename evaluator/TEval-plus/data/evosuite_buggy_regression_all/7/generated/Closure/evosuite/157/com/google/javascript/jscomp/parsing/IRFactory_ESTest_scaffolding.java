/**
 * Scaffolding file used to store all the setups needed to run 
 * tests automatically generated by EvoSuite
 * Sat Jul 29 18:24:27 GMT 2023
 */

package com.google.javascript.jscomp.parsing;

import org.evosuite.runtime.annotation.EvoSuiteClassExclude;
import org.junit.BeforeClass;
import org.junit.Before;
import org.junit.After;
import org.junit.AfterClass;
import org.evosuite.runtime.sandbox.Sandbox;
import org.evosuite.runtime.sandbox.Sandbox.SandboxMode;

@EvoSuiteClassExclude
public class IRFactory_ESTest_scaffolding {

  @org.junit.Rule 
  public org.evosuite.runtime.vnet.NonFunctionalRequirementRule nfr = new org.evosuite.runtime.vnet.NonFunctionalRequirementRule();

  private static final java.util.Properties defaultProperties = (java.util.Properties) java.lang.System.getProperties().clone(); 

  private org.evosuite.runtime.thread.ThreadStopper threadStopper =  new org.evosuite.runtime.thread.ThreadStopper (org.evosuite.runtime.thread.KillSwitchHandler.getInstance(), 3000);


  @BeforeClass 
  public static void initEvoSuiteFramework() { 
    org.evosuite.runtime.RuntimeSettings.className = "com.google.javascript.jscomp.parsing.IRFactory"; 
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
    java.lang.System.setProperty("user.dir", "/data/swf/zenodo_replication_package_new"); 
    java.lang.System.setProperty("java.io.tmpdir", "/tmp"); 
  }

  private static void initializeClasses() {
    org.evosuite.runtime.classhandling.ClassStateSupport.initializeClasses(IRFactory_ESTest_scaffolding.class.getClassLoader() ,
      "com.google.common.collect.Hashing",
      "com.google.common.collect.Sets$CartesianSet",
      "com.google.javascript.rhino.JSDocInfo$Visibility",
      "com.google.javascript.jscomp.mozilla.rhino.ast.RegExpLiteral",
      "com.google.javascript.jscomp.mozilla.rhino.ErrorReporter",
      "com.google.common.collect.Lists$RandomAccessPartition",
      "com.google.javascript.jscomp.mozilla.rhino.ast.Label",
      "com.google.common.collect.Collections2",
      "com.google.javascript.jscomp.mozilla.rhino.ast.Scope",
      "com.google.common.collect.PeekingIterator",
      "com.google.javascript.jscomp.mozilla.rhino.EvaluatorException",
      "com.google.common.collect.RegularImmutableList",
      "com.google.javascript.jscomp.mozilla.rhino.ast.FunctionCall",
      "com.google.javascript.jscomp.parsing.JsDocInfoParser",
      "com.google.common.collect.Lists$TransformingRandomAccessList",
      "com.google.javascript.jscomp.parsing.IRFactory$TransformDispatcher",
      "com.google.javascript.rhino.Node$PropListItem",
      "com.google.common.collect.RegularImmutableMap$KeySet",
      "com.google.common.collect.Sets$2",
      "com.google.common.collect.Sets$3",
      "com.google.javascript.jscomp.mozilla.rhino.ast.CatchClause",
      "com.google.javascript.jscomp.mozilla.rhino.WrappedException",
      "com.google.javascript.rhino.Node",
      "com.google.common.collect.Sets$1",
      "com.google.javascript.jscomp.mozilla.rhino.Node",
      "com.google.common.collect.ImmutableSet$ArrayImmutableSet",
      "com.google.javascript.jscomp.mozilla.rhino.ast.ArrayLiteral",
      "com.google.javascript.jscomp.parsing.IRFactory$1",
      "com.google.javascript.rhino.JSDocInfo$Marker",
      "com.google.common.collect.RegularImmutableMap",
      "com.google.javascript.jscomp.mozilla.rhino.ast.Name",
      "com.google.javascript.jscomp.mozilla.rhino.ast.DestructuringForm",
      "com.google.common.collect.RegularImmutableSet",
      "com.google.common.collect.AbstractMapEntry",
      "com.google.common.collect.Iterators$12",
      "com.google.common.collect.Iterators$11",
      "com.google.common.base.Predicate",
      "com.google.javascript.jscomp.mozilla.rhino.ast.Assignment",
      "com.google.javascript.jscomp.mozilla.rhino.ast.Block",
      "com.google.common.base.Joiner",
      "com.google.javascript.jscomp.parsing.TypeSafeDispatcher",
      "com.google.common.collect.SingletonImmutableMap",
      "com.google.javascript.rhino.jstype.JSType",
      "com.google.common.collect.ImmutableAsList",
      "com.google.javascript.rhino.Node$StringNode",
      "com.google.common.collect.Sets$SetView",
      "com.google.javascript.jscomp.mozilla.rhino.ast.InfixExpression",
      "com.google.common.collect.SingletonImmutableSet",
      "com.google.common.collect.Iterators$14",
      "com.google.common.collect.Iterators$13",
      "com.google.common.collect.RegularImmutableMap$LinkedEntry",
      "com.google.common.collect.Lists$Partition",
      "com.google.common.collect.Lists",
      "com.google.javascript.rhino.Node$SideEffectFlags",
      "com.google.common.collect.UnmodifiableListIterator",
      "com.google.javascript.jscomp.mozilla.rhino.Node$PropListItem",
      "com.google.common.collect.RegularImmutableMap$TerminalEntry",
      "com.google.javascript.jscomp.mozilla.rhino.ast.ErrorCollector",
      "com.google.common.collect.Lists$TransformingSequentialList",
      "com.google.javascript.rhino.Token",
      "com.google.javascript.rhino.Node$FileLevelJsDocBuilder",
      "com.google.common.collect.AbstractIterator",
      "com.google.common.base.Joiner$MapJoiner",
      "com.google.javascript.jscomp.mozilla.rhino.ObjToIntMap",
      "com.google.common.collect.MapDifference",
      "com.google.javascript.jscomp.mozilla.rhino.ast.Symbol",
      "com.google.common.collect.RegularImmutableMap$NonTerminalEntry",
      "com.google.common.base.Preconditions",
      "com.google.common.collect.UnmodifiableIterator",
      "com.google.javascript.rhino.JSDocInfo",
      "com.google.javascript.jscomp.parsing.Config",
      "com.google.common.collect.ImmutableEntry",
      "com.google.common.base.Joiner$1",
      "com.google.common.base.Joiner$2",
      "com.google.common.collect.ImmutableSet$TransformedImmutableSet",
      "com.google.javascript.jscomp.mozilla.rhino.ast.AstNode",
      "com.google.javascript.jscomp.mozilla.rhino.tools.ToolErrorReporter",
      "com.google.common.collect.ImmutableCollection",
      "com.google.javascript.jscomp.mozilla.rhino.ast.Jump",
      "com.google.javascript.rhino.Node$AncestorIterable",
      "com.google.common.collect.Lists$RandomAccessListWrapper",
      "com.google.common.collect.ImmutableEnumSet",
      "com.google.javascript.jscomp.mozilla.rhino.ast.Comment",
      "com.google.javascript.jscomp.mozilla.rhino.ast.ElementGet",
      "com.google.javascript.rhino.Node$NumberNode",
      "com.google.common.collect.Iterators$6",
      "com.google.common.collect.BiMap",
      "com.google.common.collect.Iterators$7",
      "com.google.common.collect.ImmutableSet",
      "com.google.common.base.Function",
      "com.google.common.collect.Lists$AbstractListWrapper",
      "com.google.common.collect.ImmutableMap",
      "com.google.common.collect.AbstractIndexedListIterator",
      "com.google.common.collect.Iterators$1",
      "com.google.common.collect.Iterators$2",
      "com.google.common.collect.Iterators$3",
      "com.google.common.collect.Sets",
      "com.google.common.collect.ImmutableCollection$EmptyImmutableCollection",
      "com.google.javascript.jscomp.mozilla.rhino.RhinoException",
      "com.google.javascript.jscomp.mozilla.rhino.ast.FunctionNode",
      "com.google.javascript.jscomp.parsing.Annotation",
      "com.google.javascript.rhino.ScriptOrFnNode",
      "com.google.common.collect.Lists$StringAsImmutableList",
      "com.google.common.collect.Lists$2",
      "com.google.javascript.jscomp.parsing.IRFactory",
      "com.google.javascript.jscomp.mozilla.rhino.ast.NodeVisitor",
      "com.google.common.collect.Lists$1",
      "com.google.javascript.jscomp.mozilla.rhino.ast.VariableDeclaration",
      "com.google.javascript.jscomp.mozilla.rhino.Node$NodeIterator",
      "com.google.javascript.jscomp.mozilla.rhino.ast.NewExpression",
      "com.google.common.collect.EmptyImmutableMap",
      "com.google.javascript.jscomp.mozilla.rhino.ast.VariableInitializer",
      "com.google.javascript.jscomp.mozilla.rhino.ast.ObjectLiteral",
      "com.google.common.collect.EmptyImmutableSet",
      "com.google.common.collect.Iterators",
      "com.google.javascript.jscomp.mozilla.rhino.ast.NumberLiteral",
      "com.google.common.collect.ImmutableList",
      "com.google.javascript.jscomp.mozilla.rhino.ast.ThrowStatement",
      "com.google.javascript.jscomp.mozilla.rhino.ast.ReturnStatement",
      "com.google.javascript.jscomp.parsing.Config$LanguageMode",
      "com.google.common.collect.RegularImmutableMap$EntrySet",
      "com.google.javascript.jscomp.mozilla.rhino.Token",
      "com.google.javascript.jscomp.mozilla.rhino.ast.AstRoot",
      "com.google.javascript.jscomp.mozilla.rhino.ast.ObjectProperty",
      "com.google.common.collect.ImmutableMap$Builder",
      "com.google.common.collect.Maps",
      "com.google.common.primitives.Ints",
      "com.google.javascript.jscomp.mozilla.rhino.ast.ScriptNode",
      "com.google.common.collect.Maps$EntryTransformer",
      "com.google.javascript.rhino.JSTypeExpression",
      "com.google.javascript.jscomp.mozilla.rhino.ast.IdeErrorReporter",
      "com.google.common.collect.RegularImmutableMap$Values"
    );
  } 

  private static void resetClasses() {
    org.evosuite.runtime.classhandling.ClassResetter.getInstance().setClassLoader(IRFactory_ESTest_scaffolding.class.getClassLoader()); 

    org.evosuite.runtime.classhandling.ClassStateSupport.resetClasses(
      "com.google.common.collect.ImmutableCollection$EmptyImmutableCollection",
      "com.google.common.collect.ImmutableCollection",
      "com.google.common.collect.ImmutableSet",
      "com.google.common.collect.Hashing",
      "com.google.common.collect.ImmutableSet$ArrayImmutableSet",
      "com.google.common.collect.RegularImmutableSet",
      "com.google.javascript.jscomp.parsing.IRFactory",
      "com.google.javascript.jscomp.parsing.TypeSafeDispatcher",
      "com.google.javascript.jscomp.parsing.IRFactory$TransformDispatcher",
      "com.google.common.base.Joiner",
      "com.google.common.base.Preconditions",
      "com.google.common.collect.Collections2",
      "com.google.javascript.jscomp.parsing.Config$LanguageMode",
      "com.google.javascript.jscomp.parsing.IRFactory$1",
      "com.google.javascript.jscomp.mozilla.rhino.Token$CommentType",
      "com.google.javascript.jscomp.mozilla.rhino.Node",
      "com.google.javascript.jscomp.mozilla.rhino.ast.AstNode",
      "com.google.javascript.jscomp.mozilla.rhino.ast.Jump",
      "com.google.javascript.jscomp.mozilla.rhino.ast.Scope",
      "com.google.javascript.jscomp.mozilla.rhino.ast.ScriptNode",
      "com.google.javascript.jscomp.mozilla.rhino.ast.AstRoot",
      "com.google.javascript.jscomp.parsing.Config",
      "com.google.common.collect.ImmutableMap",
      "com.google.common.collect.ImmutableMap$Builder",
      "com.google.common.collect.Lists",
      "com.google.common.base.Joiner$MapJoiner",
      "com.google.common.collect.Maps",
      "com.google.common.collect.AbstractMapEntry",
      "com.google.common.collect.ImmutableEntry",
      "com.google.common.collect.RegularImmutableMap",
      "com.google.common.collect.RegularImmutableMap$TerminalEntry",
      "com.google.javascript.jscomp.parsing.Annotation",
      "com.google.common.collect.RegularImmutableMap$EntrySet",
      "com.google.common.collect.UnmodifiableIterator",
      "com.google.common.collect.Iterators$1",
      "com.google.common.collect.Iterators$2",
      "com.google.common.collect.Iterators",
      "com.google.common.collect.UnmodifiableListIterator",
      "com.google.common.collect.AbstractIndexedListIterator",
      "com.google.common.collect.Iterators$11",
      "com.google.javascript.jscomp.mozilla.rhino.Kit",
      "com.google.javascript.jscomp.mozilla.rhino.ContextFactory",
      "com.google.javascript.jscomp.mozilla.rhino.ScriptableObject$Slot",
      "com.google.javascript.jscomp.mozilla.rhino.ScriptableObject",
      "com.google.javascript.jscomp.mozilla.rhino.ScriptRuntime$DefaultMessageProvider",
      "com.google.javascript.jscomp.mozilla.rhino.ScriptRuntime",
      "com.google.javascript.jscomp.mozilla.rhino.optimizer.Codegen",
      "com.google.javascript.jscomp.mozilla.rhino.Icode",
      "com.google.javascript.jscomp.mozilla.rhino.Interpreter",
      "com.google.javascript.jscomp.mozilla.rhino.Context",
      "com.google.javascript.jscomp.mozilla.rhino.jdk13.VMBridge_jdk13",
      "com.google.javascript.jscomp.mozilla.rhino.jdk15.VMBridge_jdk15",
      "com.google.javascript.jscomp.mozilla.rhino.VMBridge",
      "com.google.javascript.rhino.Node",
      "com.google.javascript.jscomp.mozilla.rhino.ast.Comment",
      "com.google.common.collect.RegularImmutableMap$NonTerminalEntry",
      "com.google.javascript.jscomp.mozilla.rhino.tools.ToolErrorReporter",
      "com.google.javascript.jscomp.mozilla.rhino.DefaultErrorReporter",
      "com.google.common.collect.Sets",
      "com.google.common.primitives.Ints",
      "com.google.javascript.rhino.Node$FileLevelJsDocBuilder",
      "com.google.javascript.jscomp.mozilla.rhino.Node$NodeIterator",
      "com.google.javascript.jscomp.mozilla.rhino.ast.Name",
      "com.google.javascript.jscomp.mozilla.rhino.ast.ErrorCollector",
      "com.google.javascript.rhino.Node$StringNode",
      "com.google.javascript.rhino.Node$NumberNode",
      "com.google.javascript.jscomp.mozilla.rhino.Token",
      "com.google.javascript.jscomp.mozilla.rhino.ast.Loop",
      "com.google.javascript.jscomp.mozilla.rhino.ast.ForLoop",
      "com.google.javascript.jscomp.mozilla.rhino.ast.XmlRef",
      "com.google.javascript.jscomp.mozilla.rhino.ast.XmlElemRef",
      "com.google.javascript.jscomp.mozilla.rhino.ContextFactory$1",
      "com.google.javascript.jscomp.mozilla.rhino.DefiningClassLoader",
      "com.google.javascript.rhino.JSDocInfo",
      "com.google.common.collect.EmptyImmutableSet",
      "com.google.javascript.jscomp.mozilla.rhino.ast.ParseProblem",
      "com.google.javascript.jscomp.mozilla.rhino.ast.ParseProblem$Type",
      "com.google.javascript.jscomp.mozilla.rhino.Node$PropListItem",
      "com.google.javascript.jscomp.mozilla.rhino.RhinoException",
      "com.google.javascript.jscomp.mozilla.rhino.EvaluatorException",
      "com.google.javascript.jscomp.mozilla.rhino.SecurityUtilities",
      "com.google.javascript.jscomp.mozilla.rhino.SecurityUtilities$1",
      "com.google.javascript.jscomp.mozilla.rhino.RhinoException$1",
      "com.google.javascript.jscomp.mozilla.rhino.WrapFactory",
      "com.google.javascript.jscomp.mozilla.rhino.IdScriptableObject",
      "com.google.javascript.jscomp.mozilla.rhino.ImporterTopLevel",
      "com.google.javascript.jscomp.mozilla.rhino.ObjArray",
      "com.google.javascript.jscomp.mozilla.rhino.ClassCache",
      "com.google.javascript.jscomp.mozilla.rhino.BaseFunction",
      "com.google.javascript.jscomp.mozilla.rhino.UniqueTag",
      "com.google.javascript.jscomp.mozilla.rhino.Scriptable",
      "com.google.javascript.jscomp.mozilla.rhino.IdScriptableObject$PrototypeValues",
      "com.google.javascript.jscomp.mozilla.rhino.IdFunctionObject",
      "com.google.javascript.jscomp.mozilla.rhino.NativeObject",
      "com.google.javascript.jscomp.mozilla.rhino.NativeError",
      "com.google.javascript.jscomp.mozilla.rhino.NativeGlobal",
      "com.google.javascript.jscomp.mozilla.rhino.Undefined",
      "com.google.javascript.jscomp.mozilla.rhino.NativeArray",
      "com.google.javascript.jscomp.mozilla.rhino.NativeString",
      "com.google.javascript.jscomp.mozilla.rhino.NativeBoolean",
      "com.google.javascript.jscomp.mozilla.rhino.NativeNumber",
      "com.google.javascript.jscomp.mozilla.rhino.NativeDate",
      "com.google.javascript.jscomp.mozilla.rhino.NativeMath",
      "com.google.javascript.jscomp.mozilla.rhino.NativeJSON",
      "com.google.javascript.jscomp.mozilla.rhino.NativeWith",
      "com.google.javascript.jscomp.mozilla.rhino.NativeCall",
      "com.google.javascript.jscomp.mozilla.rhino.NativeScript",
      "com.google.javascript.jscomp.mozilla.rhino.NativeIterator",
      "com.google.javascript.jscomp.mozilla.rhino.NativeGenerator",
      "com.google.javascript.jscomp.mozilla.rhino.NativeIterator$StopIteration",
      "com.google.javascript.jscomp.mozilla.rhino.xml.XMLLib$Factory",
      "com.google.javascript.jscomp.mozilla.rhino.xml.XMLLib$Factory$1",
      "com.google.javascript.jscomp.mozilla.rhino.LazilyLoadedCtor",
      "com.google.javascript.jscomp.mozilla.rhino.ScriptableObject$GetterSlot",
      "com.google.javascript.jscomp.mozilla.rhino.NativeContinuation",
      "com.google.javascript.jscomp.mozilla.rhino.BoundFunction",
      "com.google.javascript.jscomp.mozilla.rhino.ast.FunctionCall",
      "com.google.javascript.jscomp.mozilla.rhino.ast.NewExpression",
      "com.google.javascript.jscomp.mozilla.rhino.ast.DoLoop",
      "com.google.javascript.rhino.Node$PropListItem",
      "com.google.javascript.rhino.Node$AncestorIterable",
      "com.google.javascript.rhino.Token",
      "com.google.javascript.jscomp.mozilla.rhino.ast.IfStatement",
      "com.google.javascript.jscomp.mozilla.rhino.ast.ForInLoop",
      "com.google.javascript.jscomp.mozilla.rhino.ast.ArrayComprehensionLoop",
      "com.google.javascript.jscomp.mozilla.rhino.ast.SwitchCase",
      "com.google.javascript.jscomp.mozilla.rhino.ast.ErrorNode",
      "com.google.javascript.jscomp.mozilla.rhino.ast.WithStatement",
      "com.google.javascript.jscomp.mozilla.rhino.JavaAdapter",
      "com.google.javascript.jscomp.mozilla.rhino.NativeJavaObject",
      "com.google.javascript.jscomp.mozilla.rhino.JavaMembers",
      "com.google.javascript.jscomp.mozilla.rhino.ast.UnaryExpression",
      "com.google.javascript.jscomp.mozilla.rhino.ast.ThrowStatement",
      "com.google.javascript.jscomp.mozilla.rhino.ast.Block",
      "com.google.javascript.jscomp.mozilla.rhino.ast.NumberLiteral",
      "com.google.javascript.jscomp.mozilla.rhino.ast.XmlFragment",
      "com.google.javascript.jscomp.mozilla.rhino.ast.XmlString",
      "com.google.javascript.jscomp.mozilla.rhino.ast.ElementGet",
      "com.google.javascript.jscomp.mozilla.rhino.CompilerEnvirons",
      "com.google.javascript.jscomp.mozilla.rhino.Parser",
      "com.google.javascript.jscomp.mozilla.rhino.TokenStream",
      "com.google.javascript.jscomp.mozilla.rhino.ObjToIntMap",
      "com.google.javascript.jscomp.mozilla.rhino.ast.ExpressionStatement",
      "com.google.javascript.jscomp.mozilla.rhino.IRFactory",
      "com.google.javascript.jscomp.mozilla.rhino.Decompiler",
      "com.google.javascript.jscomp.mozilla.rhino.NodeTransformer",
      "com.google.javascript.jscomp.mozilla.rhino.optimizer.OptTransformer",
      "org.mozilla.classfile.ClassFileWriter",
      "org.mozilla.classfile.ConstantPool",
      "com.google.javascript.jscomp.mozilla.rhino.UintMap",
      "org.mozilla.classfile.ClassFileField",
      "org.mozilla.classfile.ClassFileMethod",
      "org.mozilla.classfile.FieldOrMethodRef",
      "com.google.javascript.jscomp.mozilla.rhino.optimizer.BodyCodegen",
      "com.google.javascript.jscomp.mozilla.rhino.SecurityController",
      "com.google.javascript.jscomp.mozilla.rhino.SecurityUtilities$2",
      "com.google.javascript.jscomp.mozilla.rhino.Delegator",
      "com.google.javascript.jscomp.mozilla.rhino.ast.XmlExpression",
      "com.google.javascript.jscomp.mozilla.rhino.ast.VariableInitializer",
      "com.google.javascript.jscomp.mozilla.rhino.ast.InfixExpression",
      "com.google.javascript.jscomp.mozilla.rhino.ast.ReturnStatement",
      "com.google.javascript.jscomp.mozilla.rhino.ast.ArrayLiteral",
      "com.google.javascript.jscomp.mozilla.rhino.ast.ConditionalExpression",
      "com.google.javascript.jscomp.mozilla.rhino.ContextFactory$1GlobalSetterImpl",
      "com.google.javascript.jscomp.mozilla.rhino.ast.ArrayComprehension"
    );
  }
}