/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:09:35 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.JSModule;
import com.google.javascript.jscomp.Normalize;
import com.google.javascript.jscomp.ProcessCommonJSModules;
import com.google.javascript.rhino.Node;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ProcessCommonJSModules_ESTest extends ProcessCommonJSModules_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ProcessCommonJSModules processCommonJSModules0 = new ProcessCommonJSModules(compiler0, "\"");
      JSModule jSModule0 = processCommonJSModules0.getModule();
      assertNull(jSModule0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("module.exports");
      ProcessCommonJSModules processCommonJSModules0 = new ProcessCommonJSModules(compiler0, "module.exports");
      processCommonJSModules0.process(node0, node0);
      assertFalse(node0.wasEmptyNode());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      // Undeclared exception!
      try { 
        ProcessCommonJSModules.toModuleName("./\"", "./\"");
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // java.net.URISyntaxException: Illegal character in path at index 2: ./\"
         //
         verifyException("com.google.javascript.jscomp.ProcessCommonJSModules", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      // Undeclared exception!
      try { 
        ProcessCommonJSModules.toModuleName("../\"", "../\"");
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // java.net.URISyntaxException: Illegal character in path at index 3: ../\"
         //
         verifyException("com.google.javascript.jscomp.ProcessCommonJSModules", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      String string0 = ProcessCommonJSModules.toModuleName("$$", "$$");
      assertEquals("module$$$", string0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ProcessCommonJSModules processCommonJSModules0 = new ProcessCommonJSModules(compiler0, "O-;Z/");
      String string0 = processCommonJSModules0.guessCJSModuleName("O-;Z/");
      assertEquals("module$", string0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("(");
      ProcessCommonJSModules processCommonJSModules0 = new ProcessCommonJSModules(compiler0, "(");
      Node node1 = new Node(37, node0);
      processCommonJSModules0.process(node0, node1);
      assertTrue(node0.isScript());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("module$YPa:ui.module$exports");
      Node node1 = Normalize.parseAndNormalizeTestCode(compiler0, "Using the debugger statement can halt your application if the user has a JavaScript debugger running.", "module$YPa:ui.module$exports");
      node0.addChildrenToFront(node1);
      ProcessCommonJSModules processCommonJSModules0 = new ProcessCommonJSModules(compiler0, "module$YPa:ui.module$exports");
      // Undeclared exception!
      try { 
        processCommonJSModules0.process(node0, node0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // ProcessCommonJSModules supports only one invocation per CompilerInput / script node
         //   Node(SCRIPT): [testcode]:1:0
         // Using the debugger statement can halt your application if the user has a JavaScript debugger running.
         //   Parent: NULL
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("=Z`\"MSSh H^eHBh8");
      ProcessCommonJSModules processCommonJSModules0 = new ProcessCommonJSModules(compiler0, "74yW1,yS}s=,", false);
      processCommonJSModules0.process(node0, node0);
      assertTrue(node0.isScript());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("exports");
      ProcessCommonJSModules processCommonJSModules0 = new ProcessCommonJSModules(compiler0, "exports");
      processCommonJSModules0.process(node0, node0);
      assertFalse(node0.isSetterDef());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("(");
      ProcessCommonJSModules processCommonJSModules0 = new ProcessCommonJSModules(compiler0, "(");
      processCommonJSModules0.process(node0, node0);
      node0.setSourceFileForTesting("./");
      processCommonJSModules0.process(node0, node0);
      assertEquals(1, compiler0.getErrorCount());
  }
}