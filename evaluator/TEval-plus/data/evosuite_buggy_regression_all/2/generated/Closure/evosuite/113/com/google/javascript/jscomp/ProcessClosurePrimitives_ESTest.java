/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:36:00 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.CheckLevel;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.CompilerOptions;
import com.google.javascript.jscomp.NodeTraversal;
import com.google.javascript.jscomp.PreprocessorSymbolTable;
import com.google.javascript.jscomp.ProcessClosurePrimitives;
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
      Node.newString(8, "goog.base", 1598, (-1));
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("goog.base", "goog.base");
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
      Node node0 = new Node(33);
      PreprocessorSymbolTable preprocessorSymbolTable0 = new PreprocessorSymbolTable(node0);
      CheckLevel checkLevel0 = CheckLevel.WARNING;
      ProcessClosurePrimitives processClosurePrimitives0 = new ProcessClosurePrimitives(compiler0, preprocessorSymbolTable0, checkLevel0);
      // Undeclared exception!
      try { 
        processClosurePrimitives0.hotSwapScript(node0, node0);
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
  public void test2()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      ProcessClosurePrimitives processClosurePrimitives0 = new ProcessClosurePrimitives(compiler0, (PreprocessorSymbolTable) null, compilerOptions0.brokenClosureRequiresLevel);
      Set<String> set0 = processClosurePrimitives0.getExportedVariableNames();
      assertEquals(0, set0.size());
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      Node node0 = new Node(1092);
      PreprocessorSymbolTable preprocessorSymbolTable0 = new PreprocessorSymbolTable(node0);
      ProcessClosurePrimitives processClosurePrimitives0 = new ProcessClosurePrimitives(compiler0, preprocessorSymbolTable0, compilerOptions0.checkUnreachableCode);
      processClosurePrimitives0.process(node0, node0);
      assertEquals(36, Node.QUOTED_PROP);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(86);
      Node node1 = new Node(37, node0, node0, 4095, 56);
      PreprocessorSymbolTable preprocessorSymbolTable0 = new PreprocessorSymbolTable(node0);
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      ProcessClosurePrimitives processClosurePrimitives0 = new ProcessClosurePrimitives(compiler0, preprocessorSymbolTable0, checkLevel0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, processClosurePrimitives0);
      processClosurePrimitives0.visit(nodeTraversal0, node1, node1);
      assertEquals(2, Node.POST_FLAG);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(105);
      PreprocessorSymbolTable preprocessorSymbolTable0 = new PreprocessorSymbolTable(node0);
      CheckLevel checkLevel0 = CheckLevel.WARNING;
      ProcessClosurePrimitives processClosurePrimitives0 = new ProcessClosurePrimitives(compiler0, preprocessorSymbolTable0, checkLevel0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, processClosurePrimitives0);
      Node node1 = new Node(4095, node0, 46, 4095);
      processClosurePrimitives0.visit(nodeTraversal0, node0, node1);
      assertFalse(node0.isNE());
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node.newString(105, "com.google.comon.collect.ForwardingSetMultimap", 105, 105);
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("com.google.comon.collect.ForwardingSetMultimap", "com.google.comon.collect.ForwardingSetMultimap");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: author=NOT_IMPLEMENTED and author=AUTHOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(86);
      PreprocessorSymbolTable preprocessorSymbolTable0 = new PreprocessorSymbolTable(node0);
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      ProcessClosurePrimitives processClosurePrimitives0 = new ProcessClosurePrimitives(compiler0, preprocessorSymbolTable0, checkLevel0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, processClosurePrimitives0);
      processClosurePrimitives0.visit(nodeTraversal0, node0, node0);
      assertFalse(node0.isObjectLit());
  }
}
