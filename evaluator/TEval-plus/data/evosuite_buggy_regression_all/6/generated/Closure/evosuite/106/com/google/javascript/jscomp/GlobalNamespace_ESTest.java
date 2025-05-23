/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:00:54 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.AbstractCompiler;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.GlobalNamespace;
import com.google.javascript.jscomp.MakeDeclaredNamesUnique;
import com.google.javascript.jscomp.NodeTraversal;
import com.google.javascript.jscomp.Scope;
import com.google.javascript.rhino.Node;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class GlobalNamespace_ESTest extends GlobalNamespace_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Node.newString("^9!y?N");
      GlobalNamespace globalNamespace0 = new GlobalNamespace(compiler0, node0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("com.gogle.common.collect.RegularImmutableMap$alues");
      GlobalNamespace globalNamespace0 = new GlobalNamespace(compiler0, node0, node0);
      List<GlobalNamespace.Name> list0 = globalNamespace0.getNameForest();
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("com.gogle.common.collect.RegularImmutableMap$alues");
      GlobalNamespace globalNamespace0 = new GlobalNamespace(compiler0, node0, node0);
      Scope scope0 = new Scope(node0, compiler0);
      TreeSet<Node> treeSet0 = new TreeSet<Node>();
      globalNamespace0.scanNewNodes(scope0, treeSet0);
      assertTrue(scope0.isGlobal());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("com.google.javascript.jscomp.GlobalNamespace$Name$Type", (GlobalNamespace.Name) null, false);
      String string0 = globalNamespace_Name0.toString();
      assertEquals("com.google.javascript.jscomp.GlobalNamespace$Name$Type (OTHER): globalSets=0, localSets=0, totalGets=0, aliasingGets=0, callGets=0", string0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type0 = GlobalNamespace.Ref.Type.ALIASING_GET;
      MakeDeclaredNamesUnique makeDeclaredNamesUnique0 = new MakeDeclaredNamesUnique();
      NodeTraversal nodeTraversal0 = new NodeTraversal((AbstractCompiler) null, makeDeclaredNamesUnique0);
      GlobalNamespace.Ref globalNamespace_Ref0 = null;
      try {
        globalNamespace_Ref0 = new GlobalNamespace.Ref(nodeTraversal0, (Node) null, globalNamespace_Ref_Type0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.NodeTraversal", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type0 = GlobalNamespace.Ref.Type.ALIASING_GET;
      GlobalNamespace.Ref globalNamespace_Ref0 = GlobalNamespace.Ref.createRefForTesting(globalNamespace_Ref_Type0);
      GlobalNamespace.Ref globalNamespace_Ref1 = globalNamespace_Ref0.getTwin();
      assertNull(globalNamespace_Ref1);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(40);
      GlobalNamespace globalNamespace0 = new GlobalNamespace(compiler0, node0, node0);
      globalNamespace0.getNameIndex();
      List<GlobalNamespace.Name> list0 = globalNamespace0.getNameForest();
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(40);
      GlobalNamespace globalNamespace0 = new GlobalNamespace(compiler0, node0, node0);
      globalNamespace0.getNameForest();
      Map<String, GlobalNamespace.Name> map0 = globalNamespace0.getNameIndex();
      assertEquals(0, map0.size());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Node.newString("CALL_GET");
      GlobalNamespace globalNamespace0 = new GlobalNamespace(compiler0, (Node) null, node0);
      Map<String, GlobalNamespace.Name> map0 = globalNamespace0.getNameIndex();
      assertEquals(0, map0.size());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("assign_bior");
      GlobalNamespace globalNamespace0 = new GlobalNamespace(compiler0, node0, node0);
      List<GlobalNamespace.Name> list0 = globalNamespace0.getNameForest();
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(40);
      Node node1 = new Node(64, node0, node0, node0, node0);
      GlobalNamespace globalNamespace0 = new GlobalNamespace(compiler0, node0, node1);
      Map<String, GlobalNamespace.Name> map0 = globalNamespace0.getNameIndex();
      assertEquals(0, map0.size());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(38);
      GlobalNamespace globalNamespace0 = new GlobalNamespace(compiler0, node0, node0);
      // Undeclared exception!
      try { 
        globalNamespace0.getNameForest();
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Node.newString("SeLL_GET");
      Node node1 = new Node(33, node0, node0, 1, (-3));
      GlobalNamespace globalNamespace0 = new GlobalNamespace(compiler0, node1, node0);
      Map<String, GlobalNamespace.Name> map0 = globalNamespace0.getNameIndex();
      assertTrue(map0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("xpZxw3!", (GlobalNamespace.Name) null, true);
      globalNamespace_Name0.addProperty("xpZxw3!", true);
      GlobalNamespace.Name globalNamespace_Name1 = globalNamespace_Name0.addProperty("xpZxw3!", false);
      assertNotNull(globalNamespace_Name1);
      assertNotSame(globalNamespace_Name1, globalNamespace_Name0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name(" H<!Spc{(", (GlobalNamespace.Name) null, false);
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type0 = GlobalNamespace.Ref.Type.DIRECT_GET;
      GlobalNamespace.Ref globalNamespace_Ref0 = GlobalNamespace.Ref.createRefForTesting(globalNamespace_Ref_Type0);
      globalNamespace_Name0.addRef(globalNamespace_Ref0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("", (GlobalNamespace.Name) null, true);
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type0 = GlobalNamespace.Ref.Type.ALIASING_GET;
      GlobalNamespace.Ref globalNamespace_Ref0 = GlobalNamespace.Ref.createRefForTesting(globalNamespace_Ref_Type0);
      globalNamespace_Name0.addRef(globalNamespace_Ref0);
      globalNamespace_Name0.removeRef(globalNamespace_Ref0);
      globalNamespace_Name0.removeRef(globalNamespace_Ref0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("xpZxw3!", (GlobalNamespace.Name) null, true);
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type0 = GlobalNamespace.Ref.Type.SET_FROM_GLOBAL;
      GlobalNamespace.Ref globalNamespace_Ref0 = GlobalNamespace.Ref.createRefForTesting(globalNamespace_Ref_Type0);
      globalNamespace_Name0.declaration = globalNamespace_Ref0;
      globalNamespace_Name0.addRef(globalNamespace_Name0.declaration);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("xpZxw3!", (GlobalNamespace.Name) null, true);
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type0 = GlobalNamespace.Ref.Type.SET_FROM_GLOBAL;
      GlobalNamespace.Ref globalNamespace_Ref0 = GlobalNamespace.Ref.createRefForTesting(globalNamespace_Ref_Type0);
      globalNamespace_Name0.declaration = globalNamespace_Ref0;
      globalNamespace_Name0.removeRef(globalNamespace_Ref0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("1E=G}P&=6HlBl", (GlobalNamespace.Name) null, true);
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type0 = GlobalNamespace.Ref.Type.DIRECT_GET;
      GlobalNamespace.Ref globalNamespace_Ref0 = GlobalNamespace.Ref.createRefForTesting(globalNamespace_Ref_Type0);
      globalNamespace_Name0.removeRef(globalNamespace_Ref0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("xpZxw3!", (GlobalNamespace.Name) null, true);
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type0 = GlobalNamespace.Ref.Type.ALIASING_GET;
      GlobalNamespace.Ref globalNamespace_Ref0 = GlobalNamespace.Ref.createRefForTesting(globalNamespace_Ref_Type0);
      globalNamespace_Name0.declaration = globalNamespace_Ref0;
      globalNamespace_Name0.addRefInternal(globalNamespace_Name0.declaration);
      globalNamespace_Name0.removeRef(globalNamespace_Ref0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("xpZxw3!", (GlobalNamespace.Name) null, true);
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type0 = GlobalNamespace.Ref.Type.SET_FROM_GLOBAL;
      GlobalNamespace.Ref globalNamespace_Ref0 = GlobalNamespace.Ref.createRefForTesting(globalNamespace_Ref_Type0);
      globalNamespace_Name0.addRefInternal(globalNamespace_Ref0);
      // Undeclared exception!
      try { 
        globalNamespace_Name0.removeRef(globalNamespace_Name0.declaration);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.GlobalNamespace$Name", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("com.google.javascript.jscomp.GlobalNamespace$Name$Type", (GlobalNamespace.Name) null, false);
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type0 = GlobalNamespace.Ref.Type.PROTOTYPE_GET;
      GlobalNamespace.Ref globalNamespace_Ref0 = GlobalNamespace.Ref.createRefForTesting(globalNamespace_Ref_Type0);
      globalNamespace_Name0.addRef(globalNamespace_Ref0);
      globalNamespace_Name0.removeRef(globalNamespace_Ref0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("xpZxw3!", (GlobalNamespace.Name) null, true);
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type0 = GlobalNamespace.Ref.Type.CALL_GET;
      GlobalNamespace.Ref globalNamespace_Ref0 = GlobalNamespace.Ref.createRefForTesting(globalNamespace_Ref_Type0);
      globalNamespace_Name0.addRefInternal(globalNamespace_Ref0);
      globalNamespace_Name0.removeRef(globalNamespace_Ref0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("xpZxw3!", (GlobalNamespace.Name) null, true);
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type0 = GlobalNamespace.Ref.Type.CALL_GET;
      GlobalNamespace.Ref globalNamespace_Ref0 = GlobalNamespace.Ref.createRefForTesting(globalNamespace_Ref_Type0);
      globalNamespace_Name0.addRefInternal(globalNamespace_Ref0);
      globalNamespace_Name0.addRef(globalNamespace_Ref0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("HouS1;L", (GlobalNamespace.Name) null, true);
      boolean boolean0 = globalNamespace_Name0.canEliminate();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("HouS1;L", (GlobalNamespace.Name) null, true);
      boolean boolean0 = globalNamespace_Name0.canCollapse();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("HouS1;L", (GlobalNamespace.Name) null, false);
      boolean boolean0 = globalNamespace_Name0.canCollapse();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("HouS1;L", (GlobalNamespace.Name) null, false);
      globalNamespace_Name0.setIsClassOrEnum();
      boolean boolean0 = globalNamespace_Name0.canCollapse();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("xpZxw3!", (GlobalNamespace.Name) null, true);
      GlobalNamespace.Name globalNamespace_Name1 = globalNamespace_Name0.addProperty("xpZxw3!", false);
      assertNotNull(globalNamespace_Name1);
      
      boolean boolean0 = globalNamespace_Name1.canCollapse();
      assertNotSame(globalNamespace_Name1, globalNamespace_Name0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("uKcu", (GlobalNamespace.Name) null, false);
      globalNamespace_Name0.globalSets = 221;
      boolean boolean0 = globalNamespace_Name0.canCollapse();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("", (GlobalNamespace.Name) null, false);
      globalNamespace_Name0.localSets = 1858;
      boolean boolean0 = globalNamespace_Name0.canCollapse();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("xpZxw3!", (GlobalNamespace.Name) null, true);
      globalNamespace_Name0.globalSets = (-2381);
      boolean boolean0 = globalNamespace_Name0.needsToBeStubbed();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("xpZxw3!", (GlobalNamespace.Name) null, true);
      boolean boolean0 = globalNamespace_Name0.needsToBeStubbed();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("xpZxw3!", (GlobalNamespace.Name) null, true);
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type0 = GlobalNamespace.Ref.Type.SET_FROM_LOCAL;
      GlobalNamespace.Ref globalNamespace_Ref0 = GlobalNamespace.Ref.createRefForTesting(globalNamespace_Ref_Type0);
      globalNamespace_Name0.addRef(globalNamespace_Ref0);
      boolean boolean0 = globalNamespace_Name0.needsToBeStubbed();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("UTF-8 not supported.", (GlobalNamespace.Name) null, false);
      GlobalNamespace.Name globalNamespace_Name1 = new GlobalNamespace.Name("UTF-8 not supported.", globalNamespace_Name0, false);
      globalNamespace_Name1.setIsClassOrEnum();
      boolean boolean0 = globalNamespace_Name0.isNamespace();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("", (GlobalNamespace.Name) null, true);
      boolean boolean0 = globalNamespace_Name0.isNamespace();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("xpZxw3!", (GlobalNamespace.Name) null, true);
      GlobalNamespace.Name globalNamespace_Name1 = globalNamespace_Name0.addProperty("xpZxw3!", true);
      boolean boolean0 = globalNamespace_Name1.isSimpleName();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("xpZxw3!", (GlobalNamespace.Name) null, true);
      boolean boolean0 = globalNamespace_Name0.isSimpleName();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("xpZxw3!", (GlobalNamespace.Name) null, true);
      GlobalNamespace.Name globalNamespace_Name1 = globalNamespace_Name0.addProperty("xpZxw3!", true);
      String string0 = globalNamespace_Name1.fullName();
      assertEquals("xpZxw3!.xpZxw3!", string0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("UTF-8 not supported.", (GlobalNamespace.Name) null, false);
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type0 = GlobalNamespace.Ref.Type.SET_FROM_GLOBAL;
      GlobalNamespace.Ref globalNamespace_Ref0 = GlobalNamespace.Ref.createRefForTesting(globalNamespace_Ref_Type0);
      globalNamespace_Name0.addRef(globalNamespace_Ref0);
      GlobalNamespace.Ref globalNamespace_Ref1 = globalNamespace_Name0.declaration;
      Node node0 = Node.newString((-34), "UTF-8 not supported.");
      globalNamespace_Ref1.node = node0;
      Node node1 = new Node(46, node0, node0, 47, (-3));
      GlobalNamespace.Name globalNamespace_Name1 = new GlobalNamespace.Name("UTF-8 not supported.", globalNamespace_Name0, false);
      globalNamespace_Name1.addRef(globalNamespace_Ref1);
      assertNotSame(globalNamespace_Name1, globalNamespace_Name0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type0 = GlobalNamespace.Ref.Type.SET_FROM_GLOBAL;
      GlobalNamespace.Ref globalNamespace_Ref0 = GlobalNamespace.Ref.createRefForTesting(globalNamespace_Ref_Type0);
      boolean boolean0 = globalNamespace_Ref0.isSet();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type0 = GlobalNamespace.Ref.Type.SET_FROM_LOCAL;
      GlobalNamespace.Ref globalNamespace_Ref0 = GlobalNamespace.Ref.createRefForTesting(globalNamespace_Ref_Type0);
      boolean boolean0 = globalNamespace_Ref0.isSet();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("xpZxw3!", (GlobalNamespace.Name) null, true);
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type0 = GlobalNamespace.Ref.Type.ALIASING_GET;
      GlobalNamespace.Ref globalNamespace_Ref0 = GlobalNamespace.Ref.createRefForTesting(globalNamespace_Ref_Type0);
      globalNamespace_Name0.declaration = globalNamespace_Ref0;
      boolean boolean0 = globalNamespace_Name0.declaration.isSet();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type0 = GlobalNamespace.Ref.Type.ALIASING_GET;
      GlobalNamespace.Ref globalNamespace_Ref0 = GlobalNamespace.Ref.createRefForTesting(globalNamespace_Ref_Type0);
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type1 = GlobalNamespace.Ref.Type.SET_FROM_GLOBAL;
      GlobalNamespace.Ref globalNamespace_Ref1 = globalNamespace_Ref0.cloneAndReclassify(globalNamespace_Ref_Type1);
      GlobalNamespace.Ref.markTwins(globalNamespace_Ref1, globalNamespace_Ref0);
      assertNotSame(globalNamespace_Ref1, globalNamespace_Ref0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type0 = GlobalNamespace.Ref.Type.SET_FROM_LOCAL;
      GlobalNamespace.Ref globalNamespace_Ref0 = GlobalNamespace.Ref.createRefForTesting(globalNamespace_Ref_Type0);
      // Undeclared exception!
      try { 
        GlobalNamespace.Ref.markTwins(globalNamespace_Ref0, globalNamespace_Ref0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type0 = GlobalNamespace.Ref.Type.ALIASING_GET;
      GlobalNamespace.Ref globalNamespace_Ref0 = GlobalNamespace.Ref.createRefForTesting(globalNamespace_Ref_Type0);
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type1 = GlobalNamespace.Ref.Type.SET_FROM_GLOBAL;
      GlobalNamespace.Ref globalNamespace_Ref1 = GlobalNamespace.Ref.createRefForTesting(globalNamespace_Ref_Type1);
      GlobalNamespace.Ref.markTwins(globalNamespace_Ref0, globalNamespace_Ref1);
      assertNotSame(globalNamespace_Ref0, globalNamespace_Ref1);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type0 = GlobalNamespace.Ref.Type.ALIASING_GET;
      GlobalNamespace.Ref globalNamespace_Ref0 = GlobalNamespace.Ref.createRefForTesting(globalNamespace_Ref_Type0);
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type1 = GlobalNamespace.Ref.Type.SET_FROM_LOCAL;
      GlobalNamespace.Ref globalNamespace_Ref1 = GlobalNamespace.Ref.createRefForTesting(globalNamespace_Ref_Type1);
      GlobalNamespace.Ref.markTwins(globalNamespace_Ref0, globalNamespace_Ref1);
      assertNotSame(globalNamespace_Ref0, globalNamespace_Ref1);
  }
}
