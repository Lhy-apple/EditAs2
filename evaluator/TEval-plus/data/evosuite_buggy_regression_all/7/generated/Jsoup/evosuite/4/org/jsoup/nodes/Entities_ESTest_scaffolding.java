/**
 * Scaffolding file used to store all the setups needed to run 
 * tests automatically generated by EvoSuite
 * Sat Jul 29 19:16:11 GMT 2023
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
public class Entities_ESTest_scaffolding {

  @org.junit.Rule 
  public org.evosuite.runtime.vnet.NonFunctionalRequirementRule nfr = new org.evosuite.runtime.vnet.NonFunctionalRequirementRule();

  private static final java.util.Properties defaultProperties = (java.util.Properties) java.lang.System.getProperties().clone(); 

  private org.evosuite.runtime.thread.ThreadStopper threadStopper =  new org.evosuite.runtime.thread.ThreadStopper (org.evosuite.runtime.thread.KillSwitchHandler.getInstance(), 3000);


  @BeforeClass 
  public static void initEvoSuiteFramework() { 
    org.evosuite.runtime.RuntimeSettings.className = "org.jsoup.nodes.Entities"; 
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
    org.evosuite.runtime.classhandling.ClassStateSupport.initializeClasses(Entities_ESTest_scaffolding.class.getClassLoader() ,
      "org.jsoup.select.NodeVisitor",
      "org.jsoup.nodes.Attributes",
      "org.jsoup.nodes.Evaluator$Class",
      "org.jsoup.nodes.TextNode",
      "org.jsoup.nodes.Evaluator$ContainsText",
      "org.jsoup.nodes.Evaluator$Id",
      "org.jsoup.nodes.Entities$EscapeMode",
      "org.jsoup.nodes.Evaluator$AttributeWithValueEnding",
      "org.jsoup.nodes.Evaluator$AttributeWithValueStarting",
      "org.jsoup.nodes.Evaluator$Tag",
      "org.jsoup.nodes.Document$OutputSettings",
      "org.jsoup.nodes.Evaluator$IndexEquals",
      "org.jsoup.nodes.Evaluator$Attribute",
      "org.jsoup.nodes.Element",
      "org.jsoup.nodes.Evaluator$AttributeWithValueMatching",
      "org.jsoup.nodes.Evaluator$AttributeWithValueNot",
      "org.jsoup.nodes.Evaluator$IndexEvaluator",
      "org.jsoup.nodes.Evaluator$AttributeWithValueContaining",
      "org.jsoup.helper.Validate",
      "org.jsoup.nodes.Evaluator$AttributeStarting",
      "org.jsoup.nodes.Evaluator$IndexLessThan",
      "org.jsoup.parser.Tag",
      "org.jsoup.nodes.Evaluator$AttributeWithValue",
      "org.jsoup.nodes.Node",
      "org.jsoup.nodes.Evaluator$Matches",
      "org.jsoup.nodes.Document",
      "org.jsoup.nodes.Entities",
      "org.jsoup.nodes.Evaluator$AllElements",
      "org.jsoup.nodes.Evaluator$IndexGreaterThan",
      "org.jsoup.nodes.Evaluator$AttributeKeyPair",
      "org.jsoup.select.Elements",
      "org.jsoup.nodes.Evaluator"
    );
  } 

  private static void resetClasses() {
    org.evosuite.runtime.classhandling.ClassResetter.getInstance().setClassLoader(Entities_ESTest_scaffolding.class.getClassLoader()); 

    org.evosuite.runtime.classhandling.ClassStateSupport.resetClasses(
      "org.jsoup.nodes.Entities",
      "org.jsoup.nodes.Entities$EscapeMode",
      "org.jsoup.nodes.Node",
      "org.jsoup.nodes.Element",
      "org.jsoup.nodes.Document",
      "org.jsoup.helper.Validate",
      "org.jsoup.parser.Tag",
      "org.jsoup.nodes.Attributes",
      "org.jsoup.nodes.Document$OutputSettings",
      "org.jsoup.parser.Parser",
      "org.jsoup.parser.TokenQueue",
      "org.jsoup.nodes.TextNode",
      "org.jsoup.select.NodeTraversor",
      "org.jsoup.nodes.Node$OuterHtmlVisitor",
      "org.jsoup.helper.StringUtil",
      "org.jsoup.nodes.Evaluator",
      "org.jsoup.nodes.Evaluator$AttributeKeyPair",
      "org.jsoup.nodes.Evaluator$AttributeWithValueNot",
      "org.jsoup.nodes.Attribute",
      "org.jsoup.nodes.Evaluator$AttributeWithValueMatching",
      "org.jsoup.select.Collector",
      "org.jsoup.select.Elements",
      "org.jsoup.select.Collector$Accumulator",
      "org.jsoup.nodes.Evaluator$Matches",
      "org.jsoup.nodes.Evaluator$Tag",
      "org.jsoup.nodes.Comment",
      "org.jsoup.nodes.Evaluator$Id",
      "org.jsoup.nodes.Evaluator$AllElements",
      "org.jsoup.nodes.Evaluator$IndexEvaluator",
      "org.jsoup.nodes.Evaluator$IndexLessThan",
      "org.jsoup.nodes.Evaluator$AttributeWithValueEnding",
      "org.jsoup.nodes.Attributes$Dataset",
      "org.jsoup.nodes.Evaluator$AttributeStarting",
      "org.jsoup.nodes.Evaluator$ContainsText",
      "org.jsoup.nodes.Evaluator$IndexEquals",
      "org.jsoup.nodes.Evaluator$AttributeWithValueStarting",
      "org.jsoup.select.Selector",
      "org.jsoup.nodes.Evaluator$Class",
      "org.jsoup.nodes.Evaluator$IndexGreaterThan",
      "org.jsoup.nodes.Evaluator$AttributeWithValue",
      "org.jsoup.select.Selector$SelectorParseException",
      "org.jsoup.nodes.Evaluator$AttributeWithValueContaining",
      "org.jsoup.nodes.Evaluator$Attribute",
      "org.jsoup.nodes.XmlDeclaration",
      "org.jsoup.nodes.DataNode"
    );
  }
}
