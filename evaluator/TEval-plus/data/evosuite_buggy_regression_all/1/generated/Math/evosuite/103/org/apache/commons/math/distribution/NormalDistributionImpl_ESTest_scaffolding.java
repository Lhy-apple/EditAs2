/**
 * Scaffolding file used to store all the setups needed to run 
 * tests automatically generated by EvoSuite
 * Tue Sep 26 14:08:59 GMT 2023
 */

package org.apache.commons.math.distribution;

import org.evosuite.runtime.annotation.EvoSuiteClassExclude;
import org.junit.BeforeClass;
import org.junit.Before;
import org.junit.After;
import org.junit.AfterClass;
import org.evosuite.runtime.sandbox.Sandbox;
import org.evosuite.runtime.sandbox.Sandbox.SandboxMode;

@EvoSuiteClassExclude
public class NormalDistributionImpl_ESTest_scaffolding {

  @org.junit.Rule 
  public org.evosuite.runtime.vnet.NonFunctionalRequirementRule nfr = new org.evosuite.runtime.vnet.NonFunctionalRequirementRule();

  private static final java.util.Properties defaultProperties = (java.util.Properties) java.lang.System.getProperties().clone(); 

  private org.evosuite.runtime.thread.ThreadStopper threadStopper =  new org.evosuite.runtime.thread.ThreadStopper (org.evosuite.runtime.thread.KillSwitchHandler.getInstance(), 3000);


  @BeforeClass 
  public static void initEvoSuiteFramework() { 
    org.evosuite.runtime.RuntimeSettings.className = "org.apache.commons.math.distribution.NormalDistributionImpl"; 
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
    org.evosuite.runtime.classhandling.ClassStateSupport.initializeClasses(NormalDistributionImpl_ESTest_scaffolding.class.getClassLoader() ,
      "org.apache.commons.discovery.tools.DiscoverClass",
      "org.apache.commons.math.special.Erf",
      "org.apache.commons.math.MathException",
      "org.apache.commons.math.distribution.ContinuousDistribution",
      "org.apache.commons.math.ConvergenceException",
      "org.apache.commons.discovery.tools.DefaultClassHolder",
      "org.apache.commons.discovery.ResourceNameIterator",
      "org.apache.commons.math.analysis.UnivariateRealFunction",
      "org.apache.commons.math.special.Gamma$1",
      "org.apache.commons.math.analysis.UnivariateRealSolverFactoryImpl",
      "org.apache.commons.math.MaxIterationsExceededException",
      "org.apache.commons.math.special.Gamma",
      "org.apache.commons.math.util.ContinuedFraction",
      "org.apache.commons.math.distribution.Distribution",
      "org.apache.commons.math.distribution.AbstractContinuousDistribution$1",
      "org.apache.commons.math.FunctionEvaluationException",
      "org.apache.commons.math.analysis.UnivariateRealSolver",
      "org.apache.commons.math.distribution.NormalDistribution",
      "org.apache.commons.math.analysis.UnivariateRealSolverUtils",
      "org.apache.commons.math.analysis.UnivariateRealSolverImpl",
      "org.apache.commons.discovery.tools.SPInterface",
      "org.apache.commons.math.analysis.BrentSolver",
      "org.apache.commons.math.distribution.AbstractContinuousDistribution",
      "org.apache.commons.discovery.DiscoveryException",
      "org.apache.commons.math.analysis.UnivariateRealSolverFactory",
      "org.apache.commons.math.distribution.NormalDistributionImpl",
      "org.apache.commons.math.distribution.AbstractDistribution"
    );
  } 

  private static void resetClasses() {
    org.evosuite.runtime.classhandling.ClassResetter.getInstance().setClassLoader(NormalDistributionImpl_ESTest_scaffolding.class.getClassLoader()); 

    org.evosuite.runtime.classhandling.ClassStateSupport.resetClasses(
      "org.apache.commons.math.distribution.AbstractDistribution",
      "org.apache.commons.math.distribution.AbstractContinuousDistribution",
      "org.apache.commons.math.distribution.NormalDistributionImpl",
      "org.apache.commons.math.special.Gamma",
      "org.apache.commons.math.special.Erf",
      "org.apache.commons.math.distribution.AbstractContinuousDistribution$1",
      "org.apache.commons.math.analysis.UnivariateRealSolverUtils",
      "org.apache.commons.math.analysis.UnivariateRealSolverFactory",
      "org.apache.commons.discovery.tools.DiscoverClass",
      "org.apache.commons.discovery.tools.SPInterface",
      "org.apache.commons.discovery.tools.DefaultClassHolder",
      "org.apache.commons.math.analysis.UnivariateRealSolverFactoryImpl",
      "org.apache.commons.math.analysis.UnivariateRealSolverImpl",
      "org.apache.commons.math.analysis.BrentSolver"
    );
  }
}