/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:57:06 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.AbstractCompiler;
import com.google.javascript.jscomp.FunctionToBlockMutator;
import com.google.javascript.jscomp.RenameLabels;
import com.google.javascript.rhino.Node;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class FunctionToBlockMutator_ESTest extends FunctionToBlockMutator_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      RenameLabels.DefaultNameSupplier renameLabels_DefaultNameSupplier0 = new RenameLabels.DefaultNameSupplier();
      FunctionToBlockMutator functionToBlockMutator0 = new FunctionToBlockMutator((AbstractCompiler) null, renameLabels_DefaultNameSupplier0);
      Node node0 = Node.newString("Gth=");
      // Undeclared exception!
      try { 
        functionToBlockMutator0.mutate("Gth=", node0, node0, "Gth=", false, false);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.FunctionToBlockMutator", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      RenameLabels.DefaultNameSupplier renameLabels_DefaultNameSupplier0 = new RenameLabels.DefaultNameSupplier();
      FunctionToBlockMutator.LabelNameSupplier functionToBlockMutator_LabelNameSupplier0 = new FunctionToBlockMutator.LabelNameSupplier(renameLabels_DefaultNameSupplier0);
      String string0 = functionToBlockMutator_LabelNameSupplier0.get();
      assertEquals("JSCompiler_inline_label_a", string0);
  }
}
