/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:18:24 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.CheckLevel;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.NodeTraversal;
import com.google.javascript.jscomp.PreprocessorSymbolTable;
import com.google.javascript.jscomp.ProcessClosurePrimitives;
import com.google.javascript.jscomp.ScopeCreator;
import com.google.javascript.rhino.Node;
import java.util.Set;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ProcessClosurePrimitives_ESTest extends ProcessClosurePrimitives_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.parseTestCode("goog.base");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: author=NOT_IMPLEMENTED and author=AUTHOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.parseTestCode("JSC_XMODULE_REQUIRE_ERROR");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: author=NOT_IMPLEMENTED and author=AUTHOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.OFF;
      ProcessClosurePrimitives processClosurePrimitives0 = new ProcessClosurePrimitives(compiler0, (PreprocessorSymbolTable) null, checkLevel0);
      Set<String> set0 = processClosurePrimitives0.getExportedVariableNames();
      assertEquals(0, set0.size());
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.parseTestCode("d");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: author=NOT_IMPLEMENTED and author=AUTHOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Node.newString(99, "ogle.javascript");
      CheckLevel checkLevel0 = CheckLevel.WARNING;
      ProcessClosurePrimitives processClosurePrimitives0 = new ProcessClosurePrimitives(compiler0, (PreprocessorSymbolTable) null, checkLevel0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, processClosurePrimitives0, (ScopeCreator) null);
      Node node1 = new Node(37, node0, node0, node0, node0, 0, 49);
      processClosurePrimitives0.visit(nodeTraversal0, node1, node0);
      assertEquals(0, Node.SIDE_EFFECTS_ALL);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Node.newString(105, "ogle.javascript");
      CheckLevel checkLevel0 = CheckLevel.OFF;
      ProcessClosurePrimitives processClosurePrimitives0 = new ProcessClosurePrimitives(compiler0, (PreprocessorSymbolTable) null, checkLevel0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, processClosurePrimitives0, (ScopeCreator) null);
      Node node1 = new Node(37, node0, node0, node0, node0, 52, 29);
      processClosurePrimitives0.visit(nodeTraversal0, node0, node0);
      assertFalse(node0.isParamList());
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Node.newString(105, "ogle.javascript");
      CheckLevel checkLevel0 = CheckLevel.OFF;
      ProcessClosurePrimitives processClosurePrimitives0 = new ProcessClosurePrimitives(compiler0, (PreprocessorSymbolTable) null, checkLevel0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, processClosurePrimitives0, (ScopeCreator) null);
      Node node1 = new Node(132, node0, node0, node0, node0, 52, 29);
      // Undeclared exception!
      try { 
        processClosurePrimitives0.visit(nodeTraversal0, node0, node0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.ProcessClosurePrimitives", e);
      }
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.parseTestCode("com.g>ogle.javascript.rhino.jstype.SimpleSourceFile");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: author=NOT_IMPLEMENTED and author=AUTHOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Node.newString(86, "");
      Node node1 = new Node(1, node0, 55, 130);
      PreprocessorSymbolTable preprocessorSymbolTable0 = new PreprocessorSymbolTable(node1);
      CheckLevel checkLevel0 = CheckLevel.WARNING;
      ProcessClosurePrimitives processClosurePrimitives0 = new ProcessClosurePrimitives(compiler0, preprocessorSymbolTable0, checkLevel0);
      processClosurePrimitives0.process(node0, node1);
      assertEquals(47, Node.IS_DISPATCHER);
  }
}