/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:46:12 GMT 2023
 */

package com.fasterxml.jackson.databind.introspect;

import org.junit.Test;
import static org.junit.Assert.*;
import com.fasterxml.jackson.annotation.ObjectIdGenerator;
import com.fasterxml.jackson.annotation.ObjectIdGenerators;
import com.fasterxml.jackson.annotation.ObjectIdResolver;
import com.fasterxml.jackson.annotation.SimpleObjectIdResolver;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.introspect.ObjectIdInfo;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ObjectIdInfo_ESTest extends ObjectIdInfo_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Class<ObjectIdGenerators.StringIdGenerator> class0 = ObjectIdGenerators.StringIdGenerator.class;
      ObjectIdInfo objectIdInfo0 = new ObjectIdInfo("null", class0, class0);
      Class<? extends ObjectIdGenerator<?>> class1 = objectIdInfo0.getGeneratorType();
      assertFalse(objectIdInfo0.getAlwaysAsId());
      assertEquals("class com.fasterxml.jackson.annotation.ObjectIdGenerators$StringIdGenerator", class1.toString());
      assertNotNull(class1);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Class<ObjectIdGenerators.UUIDGenerator> class0 = ObjectIdGenerators.UUIDGenerator.class;
      PropertyName propertyName0 = PropertyName.construct("}~fkR]k`=z", "}~fkR]k`=z");
      Class<SimpleObjectIdResolver> class1 = SimpleObjectIdResolver.class;
      ObjectIdInfo objectIdInfo0 = new ObjectIdInfo(propertyName0, class1, class0);
      assertFalse(objectIdInfo0.getAlwaysAsId());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Class<ObjectIdGenerators.StringIdGenerator> class0 = ObjectIdGenerators.StringIdGenerator.class;
      ObjectIdInfo objectIdInfo0 = new ObjectIdInfo("null", class0, class0);
      Class<? extends ObjectIdResolver> class1 = objectIdInfo0.getResolverType();
      assertNotNull(class1);
      
      PropertyName propertyName0 = new PropertyName("null", "null");
      ObjectIdInfo objectIdInfo1 = new ObjectIdInfo(propertyName0, class1, class0, class1);
      assertEquals(1, class1.getModifiers());
      assertFalse(objectIdInfo1.getAlwaysAsId());
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Class<ObjectIdGenerators.StringIdGenerator> class0 = ObjectIdGenerators.StringIdGenerator.class;
      ObjectIdInfo objectIdInfo0 = new ObjectIdInfo(">9t!T0.cl", class0, class0);
      boolean boolean0 = objectIdInfo0.getAlwaysAsId();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      Class<ObjectIdGenerators.StringIdGenerator> class0 = ObjectIdGenerators.StringIdGenerator.class;
      ObjectIdInfo objectIdInfo0 = new ObjectIdInfo("null", class0, class0);
      PropertyName propertyName0 = objectIdInfo0.getPropertyName();
      assertFalse(objectIdInfo0.getAlwaysAsId());
      assertNotNull(propertyName0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Class<ObjectIdGenerators.StringIdGenerator> class0 = ObjectIdGenerators.StringIdGenerator.class;
      ObjectIdInfo objectIdInfo0 = new ObjectIdInfo("null", class0, class0);
      Class<?> class1 = objectIdInfo0.getScope();
      assertFalse(objectIdInfo0.getAlwaysAsId());
      assertEquals(9, class1.getModifiers());
      assertNotNull(class1);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      Class<ObjectIdGenerators.StringIdGenerator> class0 = ObjectIdGenerators.StringIdGenerator.class;
      ObjectIdInfo objectIdInfo0 = new ObjectIdInfo("null", class0, class0);
      ObjectIdInfo objectIdInfo1 = objectIdInfo0.withAlwaysAsId(true);
      assertTrue(objectIdInfo1.getAlwaysAsId());
      assertFalse(objectIdInfo0.getAlwaysAsId());
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      Class<ObjectIdGenerators.StringIdGenerator> class0 = ObjectIdGenerators.StringIdGenerator.class;
      ObjectIdInfo objectIdInfo0 = new ObjectIdInfo(">yt!T0.Kl", class0, class0);
      ObjectIdInfo objectIdInfo1 = objectIdInfo0.withAlwaysAsId(false);
      assertSame(objectIdInfo1, objectIdInfo0);
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      Class<ObjectIdGenerators.StringIdGenerator> class0 = ObjectIdGenerators.StringIdGenerator.class;
      ObjectIdInfo objectIdInfo0 = new ObjectIdInfo("(JR5zuUTjpKl{q", (Class<?>) null, class0);
      String string0 = objectIdInfo0.toString();
      assertEquals("ObjectIdInfo: propName=(JR5zuUTjpKl{q, scope=null, generatorType=com.fasterxml.jackson.annotation.ObjectIdGenerators$StringIdGenerator, alwaysAsId=false", string0);
  }

  @Test(timeout = 4000)
  public void test9()  throws Throwable  {
      Class<ObjectIdGenerators.IntSequenceGenerator> class0 = ObjectIdGenerators.IntSequenceGenerator.class;
      ObjectIdInfo objectIdInfo0 = new ObjectIdInfo("cG}Sj5b5;q!8sM)\"Vv", class0, (Class<? extends ObjectIdGenerator<?>>) null);
      String string0 = objectIdInfo0.toString();
      assertEquals("ObjectIdInfo: propName=cG}Sj5b5;q!8sM)\"Vv, scope=com.fasterxml.jackson.annotation.ObjectIdGenerators$IntSequenceGenerator, generatorType=null, alwaysAsId=false", string0);
  }
}