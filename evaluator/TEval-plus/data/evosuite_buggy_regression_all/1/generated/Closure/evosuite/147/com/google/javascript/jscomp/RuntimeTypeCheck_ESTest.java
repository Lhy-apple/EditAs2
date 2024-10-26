/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:18:10 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.RuntimeTypeCheck;
import com.google.javascript.rhino.Node;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class RuntimeTypeCheck_ESTest extends RuntimeTypeCheck_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      RuntimeTypeCheck runtimeTypeCheck0 = new RuntimeTypeCheck(compiler0, "$");
      Node node0 = compiler0.parseTestCode("$");
      // Undeclared exception!
      runtimeTypeCheck0.process(node0, node0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("function(warning, expr) {}");
      RuntimeTypeCheck runtimeTypeCheck0 = new RuntimeTypeCheck(compiler0, "function(warning, expr) {}");
      // Undeclared exception!
      try { 
        runtimeTypeCheck0.process(node0, node0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("0R3LJb]P,");
      RuntimeTypeCheck runtimeTypeCheck0 = new RuntimeTypeCheck(compiler0, "0R3LJb]P,");
      Node node1 = new Node(4, node0, node0, node0);
      // Undeclared exception!
      try { 
        runtimeTypeCheck0.process(node1, node1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        RuntimeTypeCheck.getBoilerplateCode(compiler0, (String) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }
}
