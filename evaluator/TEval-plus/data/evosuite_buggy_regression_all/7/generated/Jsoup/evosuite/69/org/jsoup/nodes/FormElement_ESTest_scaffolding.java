/**
 * Scaffolding file used to store all the setups needed to run 
 * tests automatically generated by EvoSuite
 * Sat Jul 29 19:26:26 GMT 2023
 */

package org.jsoup.nodes;

import org.evosuite.runtime.annotation.EvoSuiteClassExclude;
import org.junit.BeforeClass;
import org.junit.Before;
import org.junit.After;
import org.junit.AfterClass;
import org.evosuite.runtime.sandbox.Sandbox;
import org.evosuite.runtime.sandbox.Sandbox.SandboxMode;

@EvoSuiteClassExclude
public class FormElement_ESTest_scaffolding {

  @org.junit.Rule 
  public org.evosuite.runtime.vnet.NonFunctionalRequirementRule nfr = new org.evosuite.runtime.vnet.NonFunctionalRequirementRule();

  private static final java.util.Properties defaultProperties = (java.util.Properties) java.lang.System.getProperties().clone(); 

  private org.evosuite.runtime.thread.ThreadStopper threadStopper =  new org.evosuite.runtime.thread.ThreadStopper (org.evosuite.runtime.thread.KillSwitchHandler.getInstance(), 3000);


  @BeforeClass 
  public static void initEvoSuiteFramework() { 
    org.evosuite.runtime.RuntimeSettings.className = "org.jsoup.nodes.FormElement"; 
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
    org.evosuite.runtime.classhandling.ClassStateSupport.initializeClasses(FormElement_ESTest_scaffolding.class.getClassLoader() ,
      "org.jsoup.select.NodeVisitor",
      "org.jsoup.nodes.Document$QuirksMode",
      "org.jsoup.Connection$Response",
      "org.jsoup.select.Evaluator$AttributeWithValueStarting",
      "org.jsoup.parser.TreeBuilder",
      "org.jsoup.parser.Token$StartTag",
      "org.jsoup.nodes.Comment",
      "org.jsoup.Connection$Request",
      "org.jsoup.select.Evaluator$IndexGreaterThan",
      "org.jsoup.nodes.LeafNode",
      "org.jsoup.select.Evaluator$IndexEvaluator",
      "org.jsoup.HttpStatusException",
      "org.jsoup.parser.Parser",
      "org.jsoup.select.QueryParser",
      "org.jsoup.helper.StringUtil$1",
      "org.jsoup.nodes.Entities$CoreCharset",
      "org.jsoup.select.Evaluator$AttributeWithValueMatching",
      "org.jsoup.select.Evaluator$Matches",
      "org.jsoup.select.Selector",
      "org.jsoup.select.Evaluator$AttributeWithValueEnding",
      "org.jsoup.nodes.Element",
      "org.jsoup.helper.HttpConnection",
      "org.jsoup.select.Evaluator$Class",
      "org.jsoup.select.NodeTraversor",
      "org.jsoup.helper.HttpConnection$KeyVal",
      "org.jsoup.nodes.Entities$1",
      "org.jsoup.UncheckedIOException",
      "org.jsoup.Connection$Method",
      "org.jsoup.nodes.Node$OuterHtmlVisitor",
      "org.jsoup.helper.HttpConnection$Base",
      "org.jsoup.parser.Token",
      "org.jsoup.parser.TokenQueue",
      "org.jsoup.select.Evaluator$AttributeKeyPair",
      "org.jsoup.select.Evaluator$MatchesOwn",
      "org.jsoup.parser.ParseSettings",
      "org.jsoup.select.NodeFilter",
      "org.jsoup.parser.Tag",
      "org.jsoup.select.CombiningEvaluator$And",
      "org.jsoup.nodes.Node",
      "org.jsoup.select.Evaluator$Attribute",
      "org.jsoup.parser.Token$EndTag",
      "org.jsoup.helper.HttpConnection$Request",
      "org.jsoup.nodes.Document",
      "org.jsoup.Connection$KeyVal",
      "org.jsoup.select.Evaluator$AttributeStarting",
      "org.jsoup.select.Evaluator$ContainsOwnText",
      "org.jsoup.nodes.Entities",
      "org.jsoup.select.Evaluator$AttributeWithValueContaining",
      "org.jsoup.select.Elements",
      "org.jsoup.Jsoup",
      "org.jsoup.nodes.DataNode",
      "org.jsoup.parser.HtmlTreeBuilder",
      "org.jsoup.nodes.FormElement",
      "org.jsoup.select.Evaluator$AllElements",
      "org.jsoup.nodes.Element$1",
      "org.jsoup.select.Evaluator$IndexLessThan",
      "org.jsoup.nodes.Attributes",
      "org.jsoup.UnsupportedMimeTypeException",
      "org.jsoup.nodes.TextNode",
      "org.jsoup.select.Evaluator$AttributeWithValue",
      "org.jsoup.select.Evaluator$AttributeWithValueNot",
      "org.jsoup.nodes.Entities$EscapeMode",
      "org.jsoup.nodes.BooleanAttribute",
      "org.jsoup.parser.XmlTreeBuilder",
      "org.jsoup.SerializationException",
      "org.jsoup.nodes.Document$OutputSettings",
      "org.jsoup.select.CombiningEvaluator",
      "org.jsoup.select.CombiningEvaluator$Or",
      "org.jsoup.select.Evaluator$ContainsText",
      "org.jsoup.select.Evaluator",
      "org.jsoup.Connection$Base",
      "org.jsoup.Connection",
      "org.jsoup.select.Evaluator$Id",
      "org.jsoup.helper.StringUtil",
      "org.jsoup.internal.Normalizer",
      "org.jsoup.select.Evaluator$IndexEquals",
      "org.jsoup.helper.Validate",
      "org.jsoup.select.Selector$SelectorParseException",
      "org.jsoup.select.Collector",
      "org.jsoup.select.Collector$Accumulator",
      "org.jsoup.parser.Token$Tag",
      "org.jsoup.nodes.XmlDeclaration",
      "org.jsoup.parser.Token$TokenType",
      "org.jsoup.internal.ConstrainableInputStream",
      "org.jsoup.nodes.Attribute",
      "org.jsoup.parser.CharacterReader",
      "org.jsoup.select.Evaluator$Tag",
      "org.jsoup.nodes.Document$OutputSettings$Syntax",
      "org.jsoup.helper.HttpConnection$Response"
    );
  } 

  private static void resetClasses() {
    org.evosuite.runtime.classhandling.ClassResetter.getInstance().setClassLoader(FormElement_ESTest_scaffolding.class.getClassLoader()); 

    org.evosuite.runtime.classhandling.ClassStateSupport.resetClasses(
      "org.jsoup.nodes.Node",
      "org.jsoup.nodes.Element",
      "org.jsoup.nodes.FormElement",
      "org.jsoup.Connection$Method",
      "org.jsoup.parser.Tag",
      "org.jsoup.parser.ParseSettings",
      "org.jsoup.nodes.Attributes",
      "org.jsoup.select.Elements",
      "org.jsoup.nodes.Document",
      "org.jsoup.internal.Normalizer",
      "org.jsoup.helper.ChangeNotifyingArrayList",
      "org.jsoup.nodes.Element$NodeList",
      "org.jsoup.select.Evaluator",
      "org.jsoup.select.Evaluator$Tag",
      "org.jsoup.select.Collector",
      "org.jsoup.select.Collector$Accumulator",
      "org.jsoup.select.NodeTraversor",
      "org.jsoup.nodes.LeafNode",
      "org.jsoup.nodes.TextNode",
      "org.jsoup.Jsoup",
      "org.jsoup.helper.HttpConnection",
      "org.jsoup.helper.HttpConnection$Base",
      "org.jsoup.helper.HttpConnection$Request",
      "org.jsoup.parser.Parser",
      "org.jsoup.parser.TreeBuilder",
      "org.jsoup.parser.HtmlTreeBuilder",
      "org.jsoup.parser.Token",
      "org.jsoup.parser.Token$Tag",
      "org.jsoup.parser.Token$StartTag",
      "org.jsoup.parser.Token$TokenType",
      "org.jsoup.parser.Token$EndTag",
      "org.jsoup.helper.HttpConnection$Response",
      "org.jsoup.nodes.Comment",
      "org.jsoup.nodes.Attribute",
      "org.jsoup.nodes.Entities$1",
      "org.jsoup.nodes.DataNode",
      "org.jsoup.helper.StringUtil$1",
      "org.jsoup.helper.StringUtil",
      "org.jsoup.parser.ParseErrorList",
      "org.jsoup.parser.HtmlTreeBuilderState",
      "org.jsoup.parser.Tokeniser",
      "org.jsoup.parser.TokeniserState",
      "org.jsoup.parser.Token$Character",
      "org.jsoup.parser.Token$Doctype",
      "org.jsoup.parser.Token$Comment",
      "org.jsoup.parser.Token$EOF",
      "org.jsoup.parser.HtmlTreeBuilderState$24",
      "org.jsoup.nodes.DocumentType",
      "org.jsoup.select.Evaluator$AttributeKeyPair",
      "org.jsoup.select.Evaluator$AttributeWithValueEnding",
      "org.jsoup.select.Evaluator$Matches",
      "org.jsoup.nodes.Element$1",
      "org.jsoup.select.Evaluator$ContainsOwnText",
      "org.jsoup.select.Evaluator$Class",
      "org.jsoup.select.Selector",
      "org.jsoup.nodes.Attributes$1",
      "org.jsoup.select.Evaluator$Id",
      "org.jsoup.select.Evaluator$AttributeWithValueMatching",
      "org.jsoup.select.Evaluator$IsEmpty",
      "org.jsoup.select.Evaluator$AttributeWithValueContaining",
      "org.jsoup.select.Evaluator$IndexEvaluator",
      "org.jsoup.select.Evaluator$IndexGreaterThan",
      "org.jsoup.nodes.Node$OuterHtmlVisitor",
      "org.jsoup.nodes.Attributes$Dataset",
      "org.jsoup.select.Evaluator$ContainsText",
      "org.jsoup.select.QueryParser",
      "org.jsoup.parser.TokenQueue",
      "org.jsoup.select.CombiningEvaluator",
      "org.jsoup.select.CombiningEvaluator$And",
      "org.jsoup.select.Evaluator$IndexLessThan",
      "org.jsoup.select.Evaluator$AttributeStarting",
      "org.jsoup.select.Evaluator$MatchesOwn",
      "org.jsoup.select.Evaluator$CssNthEvaluator",
      "org.jsoup.select.Evaluator$IsNthOfType",
      "org.jsoup.select.Evaluator$IsFirstOfType",
      "org.jsoup.select.Evaluator$AttributeWithValueNot",
      "org.jsoup.select.Evaluator$AttributeWithValue",
      "org.jsoup.nodes.Node$1",
      "org.jsoup.select.Evaluator$AllElements",
      "org.jsoup.select.NodeFilter$FilterResult",
      "org.jsoup.nodes.BooleanAttribute",
      "org.jsoup.select.Evaluator$IndexEquals",
      "org.jsoup.select.Selector$SelectorParseException",
      "org.jsoup.select.Collector$FirstFinder",
      "org.jsoup.select.Evaluator$Attribute",
      "org.jsoup.select.StructuralEvaluator$Root",
      "org.jsoup.select.StructuralEvaluator",
      "org.jsoup.select.StructuralEvaluator$ImmediateParent",
      "org.jsoup.select.Evaluator$AttributeWithValueStarting",
      "org.jsoup.nodes.XmlDeclaration",
      "org.jsoup.select.StructuralEvaluator$Parent",
      "org.jsoup.select.Evaluator$IsFirstChild",
      "org.jsoup.select.Evaluator$TagEndsWith",
      "org.jsoup.select.StructuralEvaluator$ImmediatePreviousSibling",
      "org.jsoup.select.Evaluator$IsNthChild",
      "org.jsoup.select.Evaluator$IsOnlyChild",
      "org.jsoup.select.Evaluator$IsNthLastChild",
      "org.jsoup.helper.HttpConnection$KeyVal",
      "org.jsoup.select.Evaluator$IsNthLastOfType",
      "org.jsoup.select.Evaluator$IsLastChild",
      "org.jsoup.select.CombiningEvaluator$Or",
      "org.jsoup.select.StructuralEvaluator$PreviousSibling",
      "org.jsoup.select.Evaluator$IsLastOfType",
      "org.jsoup.parser.HtmlTreeBuilderState$Constants",
      "org.jsoup.select.Evaluator$IsRoot",
      "org.jsoup.SerializationException",
      "org.jsoup.select.Evaluator$IsOnlyOfType",
      "org.jsoup.helper.Validate",
      "org.jsoup.nodes.Document$OutputSettings",
      "org.jsoup.nodes.Document$OutputSettings$Syntax",
      "org.jsoup.nodes.Entities",
      "org.jsoup.parser.CharacterReader",
      "org.jsoup.nodes.Entities$EscapeMode",
      "org.jsoup.nodes.Document$QuirksMode",
      "org.jsoup.nodes.Entities$CoreCharset"
    );
  }
}