/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 05:03:42 GMT 2023
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
      CheckLevel checkLevel0 = CheckLevel.OFF;
      ProcessClosurePrimitives processClosurePrimitives0 = new ProcessClosurePrimitives(compiler0, (PreprocessorSymbolTable) null, checkLevel0);
      Node node0 = compiler0.parseTestCode("goog.base");
      processClosurePrimitives0.process(node0, node0);
      assertEquals(1, compiler0.getErrorCount());
      assertEquals(0, compiler0.getWarningCount());
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(0, 0, 0);
      PreprocessorSymbolTable preprocessorSymbolTable0 = new PreprocessorSymbolTable(node0);
      CheckLevel checkLevel0 = CheckLevel.ERROR;
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
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      ProcessClosurePrimitives processClosurePrimitives0 = new ProcessClosurePrimitives(compiler0, (PreprocessorSymbolTable) null, checkLevel0);
      Set<String> set0 = processClosurePrimitives0.getExportedVariableNames();
      assertTrue(set0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      CheckLevel checkLevel0 = CheckLevel.WARNING;
      Compiler compiler0 = new Compiler();
      ProcessClosurePrimitives processClosurePrimitives0 = new ProcessClosurePrimitives(compiler0, (PreprocessorSymbolTable) null, checkLevel0);
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("com.google.common.collect.RegularImmutableMap$EntrySet", "com.google.common.collect.RegularImmutableMap$EntrySet");
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
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      ProcessClosurePrimitives processClosurePrimitives0 = new ProcessClosurePrimitives(compiler0, (PreprocessorSymbolTable) null, checkLevel0);
      Node node0 = new Node(37);
      // Undeclared exception!
      try { 
        processClosurePrimitives0.process(node0, node0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.WARNING;
      ProcessClosurePrimitives processClosurePrimitives0 = new ProcessClosurePrimitives(compiler0, (PreprocessorSymbolTable) null, checkLevel0);
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("N=A8f", "N=A8f");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: author=NOT_IMPLEMENTED and author=AUTHOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      CheckLevel checkLevel0 = CheckLevel.WARNING;
      Compiler compiler0 = new Compiler();
      Node node0 = Node.newString(86, "qRWada\"8y@y~Oi", 106, (-650));
      PreprocessorSymbolTable preprocessorSymbolTable0 = new PreprocessorSymbolTable(node0);
      ProcessClosurePrimitives processClosurePrimitives0 = new ProcessClosurePrimitives(compiler0, preprocessorSymbolTable0, checkLevel0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, processClosurePrimitives0, (ScopeCreator) null);
      processClosurePrimitives0.visit(nodeTraversal0, node0, node0);
      assertFalse(node0.isArrayLit());
  }
}