/**
 * Scaffolding file used to store all the setups needed to run 
 * tests automatically generated by EvoSuite
 * Sat Jul 29 19:28:07 GMT 2023
 */

package org.jsoup.parser;

import org.evosuite.runtime.annotation.EvoSuiteClassExclude;
import org.junit.BeforeClass;
import org.junit.Before;
import org.junit.After;
import org.junit.AfterClass;
import org.evosuite.runtime.sandbox.Sandbox;
import org.evosuite.runtime.sandbox.Sandbox.SandboxMode;

@EvoSuiteClassExclude
public class HtmlTreeBuilderState_ESTest_scaffolding {

  @org.junit.Rule 
  public org.evosuite.runtime.vnet.NonFunctionalRequirementRule nfr = new org.evosuite.runtime.vnet.NonFunctionalRequirementRule();

  private static final java.util.Properties defaultProperties = (java.util.Properties) java.lang.System.getProperties().clone(); 

  private org.evosuite.runtime.thread.ThreadStopper threadStopper =  new org.evosuite.runtime.thread.ThreadStopper (org.evosuite.runtime.thread.KillSwitchHandler.getInstance(), 3000);


  @BeforeClass 
  public static void initEvoSuiteFramework() { 
    org.evosuite.runtime.RuntimeSettings.className = "org.jsoup.parser.HtmlTreeBuilderState"; 
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
    org.evosuite.runtime.classhandling.ClassStateSupport.initializeClasses(HtmlTreeBuilderState_ESTest_scaffolding.class.getClassLoader() ,
      "org.jsoup.parser.HtmlTreeBuilderState$18",
      "org.jsoup.parser.HtmlTreeBuilderState$19",
      "org.jsoup.parser.HtmlTreeBuilderState$16",
      "org.jsoup.parser.HtmlTreeBuilderState$17",
      "org.jsoup.parser.HtmlTreeBuilderState$14",
      "org.jsoup.parser.TreeBuilder",
      "org.jsoup.parser.Token$StartTag",
      "org.jsoup.parser.HtmlTreeBuilderState$15",
      "org.jsoup.nodes.DocumentType",
      "org.jsoup.nodes.LeafNode",
      "org.jsoup.nodes.Element",
      "org.jsoup.parser.HtmlTreeBuilderState$12",
      "org.jsoup.parser.HtmlTreeBuilderState$13",
      "org.jsoup.parser.HtmlTreeBuilderState$10",
      "org.jsoup.parser.HtmlTreeBuilderState$Constants",
      "org.jsoup.parser.HtmlTreeBuilderState$11",
      "org.jsoup.parser.Token",
      "org.jsoup.parser.Token$Tag",
      "org.jsoup.parser.Token$Character",
      "org.jsoup.parser.HtmlTreeBuilderState$2",
      "org.jsoup.parser.HtmlTreeBuilderState$1",
      "org.jsoup.nodes.Node",
      "org.jsoup.parser.HtmlTreeBuilderState$4",
      "org.jsoup.parser.HtmlTreeBuilderState$3",
      "org.jsoup.parser.Token$EndTag",
      "org.jsoup.parser.HtmlTreeBuilderState",
      "org.jsoup.parser.HtmlTreeBuilderState$23",
      "org.jsoup.parser.HtmlTreeBuilderState$9",
      "org.jsoup.parser.HtmlTreeBuilderState$24",
      "org.jsoup.parser.HtmlTreeBuilderState$21",
      "org.jsoup.parser.HtmlTreeBuilderState$22",
      "org.jsoup.parser.HtmlTreeBuilderState$6",
      "org.jsoup.parser.HtmlTreeBuilderState$5",
      "org.jsoup.parser.HtmlTreeBuilderState$20",
      "org.jsoup.parser.HtmlTreeBuilderState$8",
      "org.jsoup.parser.HtmlTreeBuilder",
      "org.jsoup.parser.HtmlTreeBuilderState$7",
      "org.jsoup.nodes.FormElement"
    );
  } 

  private static void resetClasses() {
    org.evosuite.runtime.classhandling.ClassResetter.getInstance().setClassLoader(HtmlTreeBuilderState_ESTest_scaffolding.class.getClassLoader()); 

    org.evosuite.runtime.classhandling.ClassStateSupport.resetClasses(
      "org.jsoup.parser.HtmlTreeBuilderState",
      "org.jsoup.parser.HtmlTreeBuilderState$Constants",
      "org.jsoup.parser.TokeniserState"
    );
  }
}
