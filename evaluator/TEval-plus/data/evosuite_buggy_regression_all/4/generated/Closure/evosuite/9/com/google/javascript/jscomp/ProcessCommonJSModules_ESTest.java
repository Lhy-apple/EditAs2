/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:08:10 GMT 2023
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
  public void test0()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ProcessCommonJSModules processCommonJSModules0 = new ProcessCommonJSModules(compiler0, "");
      JSModule jSModule0 = processCommonJSModules0.getModule();
      assertNull(jSModule0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("module.exports");
      ProcessCommonJSModules processCommonJSModules0 = new ProcessCommonJSModules(compiler0, "%,,9URz?<");
      processCommonJSModules0.process(node0, node0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      String string0 = ProcessCommonJSModules.toModuleName(".//", ".//");
      assertEquals("module$", string0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      String string0 = ProcessCommonJSModules.toModuleName("../", "../");
      assertEquals("module$..$..$", string0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      String string0 = ProcessCommonJSModules.toModuleName("_", "_");
      assertEquals("module$_", string0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("../");
      ProcessCommonJSModules processCommonJSModules0 = new ProcessCommonJSModules(compiler0, "../");
      node0.setSourceFileForTesting("../");
      processCommonJSModules0.process(node0, node0);
      assertEquals(29, Node.JSDOC_INFO_PROP);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ProcessCommonJSModules processCommonJSModules0 = new ProcessCommonJSModules(compiler0, "/");
      Node node0 = compiler0.parseTestCode("/");
      Node node1 = Normalize.parseAndNormalizeTestCode(compiler0, "./", "PRESERVE_TRY");
      node0.addChildToBack(node1);
      // Undeclared exception!
      try { 
        processCommonJSModules0.process(node0, node0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // ProcessCommonJSModules supports only one invocation per CompilerInput / script node
         //   Node(SCRIPT): [testcode]:-1:-1
         // [source unknown]
         //   Parent: NULL
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("%,,9Uxz?0");
      ProcessCommonJSModules processCommonJSModules0 = new ProcessCommonJSModules(compiler0, "./", false);
      processCommonJSModules0.process(node0, node0);
      assertFalse(node0.isGetProp());
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ProcessCommonJSModules processCommonJSModules0 = new ProcessCommonJSModules(compiler0, "/");
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "exports", (String) null);
      processCommonJSModules0.process(node0, node0);
      assertEquals(38, Node.SYNTHETIC_BLOCK_PROP);
  }

  @Test(timeout = 4000)
  public void test9()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("%,,9URz?<");
      ProcessCommonJSModules processCommonJSModules0 = new ProcessCommonJSModules(compiler0, "%,,9URz?<");
      processCommonJSModules0.process(node0, node0);
      node0.setSourceFileForTesting("./");
      processCommonJSModules0.process(node0, node0);
      assertEquals(1, compiler0.getErrorCount());
  }
}