/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:25:23 GMT 2023
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
      ProcessCommonJSModules processCommonJSModules0 = new ProcessCommonJSModules(compiler0, "./mYf}yQ[Hi.(N=K<Tm", false);
      JSModule jSModule0 = processCommonJSModules0.getModule();
      assertNull(jSModule0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "module.exports", "module.exports");
      ProcessCommonJSModules processCommonJSModules0 = new ProcessCommonJSModules(compiler0, "module.exports");
      processCommonJSModules0.process(node0, node0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ProcessCommonJSModules processCommonJSModules0 = new ProcessCommonJSModules(compiler0, "$cJ5PDIp^\u0004V:uMs/", false);
      String string0 = processCommonJSModules0.guessCJSModuleName("$cJ5PDIp^\u0004V:uMs/");
      assertEquals("module$", string0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      // Undeclared exception!
      try { 
        ProcessCommonJSModules.toModuleName("./Operand out of ange, bitwise operation will lose information: {0}", "Operand out of ange, bitwise operation will lose information: {0}/");
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // java.net.URISyntaxException: Illegal character in scheme name at index 7: Operand out of ange, bitwise operation will lose information: {0}/
         //
         verifyException("com.google.javascript.jscomp.ProcessCommonJSModules", e);
      }
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      // Undeclared exception!
      try { 
        ProcessCommonJSModules.toModuleName("../BF\"L{XzF*[-cnXOh", "mYf");
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // java.net.URISyntaxException: Illegal character in path at index 5: ../BF\"L{XzF*[-cnXOh
         //
         verifyException("com.google.javascript.jscomp.ProcessCommonJSModules", e);
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      String string0 = ProcessCommonJSModules.toModuleName("", "");
      assertEquals("module$", string0);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ProcessCommonJSModules processCommonJSModules0 = new ProcessCommonJSModules(compiler0, "(function(t){})(y.prototype);");
      Node[] nodeArray0 = new Node[0];
      Node node0 = new Node(37, nodeArray0);
      processCommonJSModules0.process(node0, node0);
      assertEquals(49, Node.DIRECT_EVAL);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "(function(t){})(y.prototype);", "(function(t){})(y.prototype);");
      ProcessCommonJSModules processCommonJSModules0 = new ProcessCommonJSModules(compiler0, "(function(t){})(y.prototype);");
      Node node1 = Normalize.parseAndNormalizeTestCode(compiler0, "./", "/");
      Node node2 = new Node(52, node0, node1, 29, 47);
      // Undeclared exception!
      try { 
        processCommonJSModules0.process(node0, node2);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // ProcessCommonJSModules supports only one invocation per CompilerInput / script node
         //   Node(SCRIPT): [testcode]:-1:-1
         // [source unknown]
         //   Parent(INSTANCEOF): [source unknown]
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "N0BO6\"", "N0BO6\"");
      ProcessCommonJSModules processCommonJSModules0 = new ProcessCommonJSModules(compiler0, "$cJ5PDIp^\u0004V:uMs/", false);
      processCommonJSModules0.process(node0, node0);
      assertTrue(node0.hasChildren());
  }

  @Test(timeout = 4000)
  public void test9()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ProcessCommonJSModules processCommonJSModules0 = new ProcessCommonJSModules(compiler0, "kyl");
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "exports", "require");
      processCommonJSModules0.process(node0, node0);
      assertFalse(node0.isFor());
  }
}