/**
 * Scaffolding file used to store all the setups needed to run 
 * tests automatically generated by EvoSuite
 * Tue Sep 26 23:31:17 GMT 2023
 */

package org.apache.commons.math.complex;

import org.evosuite.runtime.annotation.EvoSuiteClassExclude;
import org.junit.BeforeClass;
import org.junit.Before;
import org.junit.After;
import org.junit.AfterClass;
import org.evosuite.runtime.sandbox.Sandbox;
import org.evosuite.runtime.sandbox.Sandbox.SandboxMode;

@EvoSuiteClassExclude
public class Complex_ESTest_scaffolding {

  @org.junit.Rule 
  public org.evosuite.runtime.vnet.NonFunctionalRequirementRule nfr = new org.evosuite.runtime.vnet.NonFunctionalRequirementRule();

  private static final java.util.Properties defaultProperties = (java.util.Properties) java.lang.System.getProperties().clone(); 

  private org.evosuite.runtime.thread.ThreadStopper threadStopper =  new org.evosuite.runtime.thread.ThreadStopper (org.evosuite.runtime.thread.KillSwitchHandler.getInstance(), 3000);


  @BeforeClass 
  public static void initEvoSuiteFramework() { 
    org.evosuite.runtime.RuntimeSettings.className = "org.apache.commons.math.complex.Complex"; 
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
    org.evosuite.runtime.classhandling.ClassStateSupport.initializeClasses(Complex_ESTest_scaffolding.class.getClassLoader() ,
      "org.apache.commons.math.exception.util.Localizable",
      "org.apache.commons.math.util.FastMath$lnMant",
      "org.apache.commons.math.exception.NumberIsTooSmallException",
      "org.apache.commons.math.exception.MathIllegalArgumentException",
      "org.apache.commons.math.exception.NullArgumentException",
      "org.apache.commons.math.exception.util.ExceptionContext",
      "org.apache.commons.math.complex.Complex",
      "org.apache.commons.math.exception.MathArithmeticException",
      "org.apache.commons.math.util.FastMath",
      "org.apache.commons.math.util.MathUtils",
      "org.apache.commons.math.util.FastMath$ExpIntTable",
      "org.apache.commons.math.util.FastMathLiteralArrays",
      "org.apache.commons.math.Field",
      "org.apache.commons.math.complex.ComplexField",
      "org.apache.commons.math.complex.ComplexField$LazyHolder",
      "org.apache.commons.math.exception.util.LocalizedFormats",
      "org.apache.commons.math.exception.MathIllegalNumberException",
      "org.apache.commons.math.exception.NotFiniteNumberException",
      "org.apache.commons.math.FieldElement",
      "org.apache.commons.math.exception.util.ExceptionContextProvider",
      "org.apache.commons.math.util.FastMath$ExpFracTable",
      "org.apache.commons.math.exception.NotPositiveException",
      "org.apache.commons.math.exception.util.ArgUtils"
    );
  } 

  private static void resetClasses() {
    org.evosuite.runtime.classhandling.ClassResetter.getInstance().setClassLoader(Complex_ESTest_scaffolding.class.getClassLoader()); 

    org.evosuite.runtime.classhandling.ClassStateSupport.resetClasses(
      "org.apache.commons.math.complex.Complex",
      "org.apache.commons.math.util.FastMath",
      "org.apache.commons.math.util.FastMathLiteralArrays",
      "org.apache.commons.math.util.FastMath$ExpIntTable",
      "org.apache.commons.math.util.FastMathResources",
      "org.apache.commons.math.util.FastMath$ExpFracTable",
      "org.apache.commons.math.util.FastMathCalc",
      "org.apache.commons.math.util.FastMath$lnMant",
      "org.apache.commons.math.exception.util.LocalizedFormats",
      "org.apache.commons.math.complex.ComplexField",
      "org.apache.commons.math.complex.ComplexField$LazyHolder",
      "org.apache.commons.math.util.MathUtils",
      "org.apache.commons.math.exception.MathIllegalArgumentException",
      "org.apache.commons.math.exception.MathIllegalNumberException",
      "org.apache.commons.math.exception.NumberIsTooSmallException",
      "org.apache.commons.math.exception.NotPositiveException",
      "org.apache.commons.math.exception.util.ExceptionContext",
      "org.apache.commons.math.exception.util.ArgUtils",
      "org.apache.commons.math.exception.NullArgumentException"
    );
  }
}
