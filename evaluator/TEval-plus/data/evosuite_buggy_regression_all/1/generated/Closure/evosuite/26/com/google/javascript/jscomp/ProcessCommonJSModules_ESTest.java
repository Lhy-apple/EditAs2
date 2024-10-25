/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:05:23 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.AbstractCompiler;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.JSModule;
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
      Node node0 = compiler0.parseTestCode("exports");
      ProcessCommonJSModules processCommonJSModules0 = new ProcessCommonJSModules(compiler0, "exports");
      processCommonJSModules0.process(node0, node0);
      assertFalse(node0.isExprResult());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      ProcessCommonJSModules processCommonJSModules0 = new ProcessCommonJSModules((AbstractCompiler) null, "?E-Y~i5>.80<`T");
      JSModule jSModule0 = processCommonJSModules0.getModule();
      assertNull(jSModule0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("module.exports");
      ProcessCommonJSModules processCommonJSModules0 = new ProcessCommonJSModules(compiler0, "module.exports");
      processCommonJSModules0.process(node0, node0);
      assertFalse(node0.isFromExterns());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ProcessCommonJSModules processCommonJSModules0 = new ProcessCommonJSModules(compiler0, "AEL#fYbIe/");
      String string0 = processCommonJSModules0.guessCJSModuleName("AEL#fYbIe/");
      assertEquals("module$", string0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      String string0 = ProcessCommonJSModules.toModuleName("./modle$", "./modle$");
      assertEquals("module$modle$", string0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      String string0 = ProcessCommonJSModules.toModuleName("../com.google.javascript.jscomp.SimpleRegion/", "../com.google.javascript.jscomp.SimpleRegion/");
      assertEquals("module$..$com.google.javascript.jscomp.SimpleRegion$", string0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      String string0 = ProcessCommonJSModules.toModuleName("module.exports", "module.exports");
      assertEquals("module$module.exports", string0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ProcessCommonJSModules processCommonJSModules0 = new ProcessCommonJSModules(compiler0, "require");
      Node node0 = new Node(37);
      processCommonJSModules0.process(node0, node0);
      assertEquals(54, Node.SLASH_V);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode(")7m");
      Node node1 = compiler0.parseSyntheticCode(")7m");
      node0.addChildToBack(node1);
      ProcessCommonJSModules processCommonJSModules0 = new ProcessCommonJSModules(compiler0, ")7m");
      // Undeclared exception!
      try { 
        processCommonJSModules0.process(node0, node0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // ProcessCommonJSModules supports only one invocation per CompilerInput / script node
         //   Node(SCRIPT):  [synthetic:1] :-1:-1
         // [source unknown]
         //   Parent: NULL
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode(".js$");
      ProcessCommonJSModules processCommonJSModules0 = new ProcessCommonJSModules(compiler0, ".js$", false);
      processCommonJSModules0.process(node0, node0);
      assertFalse(node0.isReturn());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("provide");
      ProcessCommonJSModules processCommonJSModules0 = new ProcessCommonJSModules(compiler0, "provide");
      processCommonJSModules0.process(node0, node0);
      node0.setSourceFileForTesting("./");
      processCommonJSModules0.process(node0, node0);
      assertFalse(compiler0.hasErrors());
      assertEquals(0, compiler0.getErrorCount());
  }
}
