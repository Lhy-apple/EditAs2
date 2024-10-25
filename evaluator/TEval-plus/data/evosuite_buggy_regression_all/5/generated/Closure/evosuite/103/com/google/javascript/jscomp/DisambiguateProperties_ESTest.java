/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:40:19 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.common.collect.Multimap;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.ConcreteType;
import com.google.javascript.jscomp.DisambiguateProperties;
import com.google.javascript.jscomp.TightenTypes;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.ObjectType;
import java.util.Collection;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DisambiguateProperties_ESTest extends DisambiguateProperties_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      DisambiguateProperties<JSType> disambiguateProperties0 = DisambiguateProperties.forJSTypeSystem(compiler0);
      // Undeclared exception!
      try { 
        disambiguateProperties0.process((Node) null, (Node) null);
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
  public void test1()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Node.newString(33, "dWDu#", 33, 33);
      TightenTypes tightenTypes0 = new TightenTypes(compiler0);
      Node node1 = Node.newString("dWDu#");
      node0.addChildrenToBack(node1);
      DisambiguateProperties<ConcreteType> disambiguateProperties0 = DisambiguateProperties.forConcreteTypeSystem(compiler0, tightenTypes0);
      disambiguateProperties0.process(node0, node0);
      assertEquals(4095, Node.COLUMN_MASK);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      TightenTypes tightenTypes0 = new TightenTypes(compiler0);
      DisambiguateProperties<ConcreteType> disambiguateProperties0 = DisambiguateProperties.forConcreteTypeSystem(compiler0, tightenTypes0);
      ConcreteType concreteType0 = ConcreteType.ALL;
      ConcreteType concreteType1 = disambiguateProperties0.getTypeWithProperty("TightenTypes pass appears to be stuck in an infinite loop.", concreteType0);
      assertNull(concreteType1);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Node.newString(33, "dWDu#", 33, 33);
      TightenTypes tightenTypes0 = new TightenTypes(compiler0);
      Node node1 = Node.newString("dWDu#");
      node0.addChildrenToBack(node1);
      DisambiguateProperties<ConcreteType> disambiguateProperties0 = DisambiguateProperties.forConcreteTypeSystem(compiler0, tightenTypes0);
      disambiguateProperties0.process(node1, node0);
      assertEquals(23, Node.VARIABLE_PROP);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Node.newString(64, "Stuck in loop expanding types to skip.", 64, 64);
      TightenTypes tightenTypes0 = new TightenTypes(compiler0);
      DisambiguateProperties<ConcreteType> disambiguateProperties0 = DisambiguateProperties.forConcreteTypeSystem(compiler0, tightenTypes0);
      disambiguateProperties0.process(node0, node0);
      assertFalse(node0.isQuotedString());
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      DisambiguateProperties<ConcreteType> disambiguateProperties0 = DisambiguateProperties.forConcreteTypeSystem(compiler0, (TightenTypes) null);
      Multimap<String, Collection<ConcreteType>> multimap0 = disambiguateProperties0.getRenamedTypesForTesting();
      assertNotNull(multimap0);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      TightenTypes tightenTypes0 = new TightenTypes(compiler0);
      DisambiguateProperties<ConcreteType> disambiguateProperties0 = DisambiguateProperties.forConcreteTypeSystem(compiler0, tightenTypes0);
      ConcreteType.ConcreteInstanceType concreteType_ConcreteInstanceType0 = new ConcreteType.ConcreteInstanceType(tightenTypes0, (ObjectType) null);
      // Undeclared exception!
      try { 
        disambiguateProperties0.getTypeWithProperty("d^.g3j]UwbfFr)r,", concreteType_ConcreteInstanceType0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.TightenTypes", e);
      }
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      TightenTypes tightenTypes0 = new TightenTypes(compiler0);
      DisambiguateProperties<ConcreteType> disambiguateProperties0 = DisambiguateProperties.forConcreteTypeSystem(compiler0, tightenTypes0);
      ConcreteType concreteType0 = ConcreteType.NONE;
      ConcreteType concreteType1 = disambiguateProperties0.getTypeWithProperty("TightenTypes pass appears to be stuck in an infinite loop.", concreteType0);
      assertNotNull(concreteType1);
  }
}
