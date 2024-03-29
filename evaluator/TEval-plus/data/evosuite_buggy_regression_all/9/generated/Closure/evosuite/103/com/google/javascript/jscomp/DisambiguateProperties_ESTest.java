/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 05:01:59 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import com.google.common.collect.Multimap;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.ConcreteType;
import com.google.javascript.jscomp.DisambiguateProperties;
import com.google.javascript.jscomp.TightenTypes;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.jstype.JSType;
import java.util.ArrayDeque;
import java.util.Collection;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DisambiguateProperties_ESTest extends DisambiguateProperties_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Node.newNumber(885.2);
      DisambiguateProperties<JSType> disambiguateProperties0 = DisambiguateProperties.forJSTypeSystem(compiler0);
      disambiguateProperties0.process(node0, node0);
      assertEquals(1, Node.DECR_FLAG);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("BP", "com.google.javascript.jscomp.DisambiguateProperties$FindExternProperties");
      TightenTypes tightenTypes0 = new TightenTypes(compiler0);
      DisambiguateProperties<ConcreteType> disambiguateProperties0 = DisambiguateProperties.forConcreteTypeSystem(compiler0, tightenTypes0);
      disambiguateProperties0.process(node0, node0);
      disambiguateProperties0.process(node0, node0);
      assertEquals(41, Node.BRACELESS_TYPE);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      TightenTypes tightenTypes0 = new TightenTypes(compiler0);
      ConcreteType concreteType0 = ConcreteType.ALL;
      DisambiguateProperties<ConcreteType> disambiguateProperties0 = DisambiguateProperties.forConcreteTypeSystem(compiler0, tightenTypes0);
      ConcreteType concreteType1 = disambiguateProperties0.getTypeWithProperty("TightenTypes pass appears to be stuck in an infinite loop.", concreteType0);
      assertNull(concreteType1);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("BP", "com.google.javascript.jscomp.DisambiguateProperties$FindExternProperties");
      TightenTypes tightenTypes0 = new TightenTypes(compiler0);
      Node node1 = compiler0.parseSyntheticCode("_", "DisambiguateProperties$FindExternProperties");
      DisambiguateProperties<ConcreteType> disambiguateProperties0 = DisambiguateProperties.forConcreteTypeSystem(compiler0, tightenTypes0);
      disambiguateProperties0.process(node1, node0);
      assertEquals(35, Node.QUOTED_PROP);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      TightenTypes tightenTypes0 = new TightenTypes(compiler0);
      DisambiguateProperties<ConcreteType> disambiguateProperties0 = DisambiguateProperties.forConcreteTypeSystem(compiler0, tightenTypes0);
      Multimap<String, Collection<ConcreteType>> multimap0 = disambiguateProperties0.getRenamedTypesForTesting();
      assertNotNull(multimap0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ArrayDeque<ConcreteType> arrayDeque0 = new ArrayDeque<ConcreteType>();
      ConcreteType concreteType0 = ConcreteType.createForTypes(arrayDeque0);
      TightenTypes tightenTypes0 = new TightenTypes(compiler0);
      DisambiguateProperties<ConcreteType> disambiguateProperties0 = DisambiguateProperties.forConcreteTypeSystem(compiler0, tightenTypes0);
      ConcreteType concreteType1 = disambiguateProperties0.getTypeWithProperty("?jj{gcm7k\"D;7+:", concreteType0);
      assertNotNull(concreteType1);
  }
}
