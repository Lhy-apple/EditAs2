/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 02:21:36 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.Normalize;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.SimpleErrorReporter;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Normalize_ESTest extends Normalize_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("l:ET");
      Normalize.VerifyConstants normalize_VerifyConstants0 = new Normalize.VerifyConstants(compiler0, true);
      // Undeclared exception!
      try { 
        normalize_VerifyConstants0.process(node0, node0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("l:ET");
      Node node1 = new Node(17, node0, node0, node0, node0, 35, 33);
      Normalize.VerifyConstants normalize_VerifyConstants0 = new Normalize.VerifyConstants(compiler0, true);
      normalize_VerifyConstants0.process(node0, node0);
      assertEquals(17, Node.TYPE_PROP);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("l:ET");
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      JSType[] jSTypeArray0 = new JSType[3];
      Node node1 = jSTypeRegistry0.createParameters(jSTypeArray0);
      Node node2 = new Node((-1419), node0, node0, node1, node1, 0, 1255);
      Normalize.VerifyConstants normalize_VerifyConstants0 = new Normalize.VerifyConstants(compiler0, true);
      normalize_VerifyConstants0.process(node1, node1);
      assertFalse(node1.isSyntheticBlock());
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("l:ET");
      Node node1 = new Node(17, node0, node0, node0, node0, 35, 33);
      Normalize.VerifyConstants normalize_VerifyConstants0 = new Normalize.VerifyConstants(compiler0, false);
      normalize_VerifyConstants0.process(node0, node0);
      assertEquals(1, Node.PROPERTY_FLAG);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("com.google.protobuf.Descriptors$ServiceDescriptor");
      Node node1 = new Node(113, node0, node0, node0, node0, 49, 27);
      Normalize normalize0 = new Normalize(compiler0, true);
      // Undeclared exception!
      try { 
        normalize0.process(node1, node1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // Normalize constraints violated:
         // WHILE node
         //   Node(FOR):  [testcode] :49:27
         // [source unknown]
         //   Parent: NULL
         //
         verifyException("com.google.javascript.jscomp.Normalize", e);
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("Qn-aE");
      Node node1 = new Node(105, node0, node0, node0, node0, 105, (-1));
      Normalize normalize0 = new Normalize(compiler0, false);
      // Undeclared exception!
      try { 
        normalize0.process(node1, node1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // null
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("l:ET");
      Node node1 = new Node(17, node0, node0, node0, node0, 35, 33);
      Normalize normalize0 = new Normalize(compiler0, false);
      normalize0.process(node0, node0);
      // Undeclared exception!
      try { 
        normalize0.process(node0, node1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // null
         //   Node(SCRIPT):  [testcode] :1:0
         // [source unknown]
         //   Parent(GE):  [testcode] :35:33
         // [source unknown]
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("lQn-aE%");
      Node node1 = new Node(105, node0, node0, node0, node0, 105, (-1));
      Normalize normalize0 = new Normalize(compiler0, false);
      // Undeclared exception!
      try { 
        normalize0.process(node0, node1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // null
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }
}