/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:24:31 GMT 2023
 */

package com.fasterxml.jackson.databind.type;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.Module;
import com.fasterxml.jackson.databind.node.DecimalNode;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.ReferenceType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.util.HashMap;
import java.util.LinkedList;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.testdata.EvoSuiteFile;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class SimpleType_ESTest extends SimpleType_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      ReferenceType referenceType0 = ReferenceType.upgradeFrom(simpleType0, simpleType0);
      ReferenceType referenceType1 = referenceType0.withContentValueHandler(simpleType0);
      assertFalse(referenceType1.isInterface());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_BOOL;
      // Undeclared exception!
      try { 
        simpleType0.withContentValueHandler((Object) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Simple types have no content types; can not call withContenValueHandler()
         //
         verifyException("com.fasterxml.jackson.databind.type.SimpleType", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_BOOL;
      // Undeclared exception!
      try { 
        simpleType0.withContentTypeHandler("Z;");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Simple types have no content types; can not call withContenTypeHandler()
         //
         verifyException("com.fasterxml.jackson.databind.type.SimpleType", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_INT;
      String string0 = simpleType0.toString();
      assertEquals("[simple type, class int]", string0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_ENUM;
      boolean boolean0 = simpleType0.isContainerType();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<HashMap> class0 = HashMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      SimpleType simpleType0 = new SimpleType(mapType0);
      SimpleType simpleType1 = simpleType0.withTypeHandler(mapType0);
      assertNotSame(simpleType1, simpleType0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_INT;
      MapType mapType0 = new MapType(simpleType0, simpleType0, simpleType0);
      // Undeclared exception!
      try { 
        simpleType0.withContentType(mapType0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Simple types have no content types; can not call withContentType()
         //
         verifyException("com.fasterxml.jackson.databind.type.SimpleType", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_INT;
      String string0 = simpleType0.getErasedSignature();
      assertEquals("I", string0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Class<SimpleType> class0 = SimpleType.class;
      SimpleType simpleType0 = new SimpleType(class0);
      assertEquals(0, simpleType0.containedTypeCount());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Class<HashMap> class0 = HashMap.class;
      // Undeclared exception!
      try { 
        SimpleType.construct(class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not construct SimpleType for a Map (class: java.util.HashMap)
         //
         verifyException("com.fasterxml.jackson.databind.type.SimpleType", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Class<LinkedList> class0 = LinkedList.class;
      // Undeclared exception!
      try { 
        SimpleType.construct(class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not construct SimpleType for a Collection (class: java.util.LinkedList)
         //
         verifyException("com.fasterxml.jackson.databind.type.SimpleType", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_BOOL;
      JavaType javaType0 = simpleType0._narrow(class0);
      assertFalse(javaType0.isPrimitive());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      JavaType javaType0 = simpleType0._narrow(class0);
      assertSame(javaType0, simpleType0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_BOOL;
      SimpleType simpleType1 = simpleType0.withTypeHandler((Object) null);
      assertSame(simpleType1, simpleType0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Class<Module> class0 = Module.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      SimpleType simpleType1 = simpleType0.withValueHandler((Object) null);
      assertSame(simpleType1, simpleType0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Class<String> class0 = String.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      SimpleType simpleType1 = simpleType0.withStaticTyping();
      SimpleType simpleType2 = simpleType1.withStaticTyping();
      assertTrue(simpleType2.useStaticType());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<HashMap> class0 = HashMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      SimpleType simpleType0 = new SimpleType(mapType0);
      String string0 = simpleType0.toCanonical();
      assertEquals("java.util.HashMap<java.lang.Object,java.lang.Object>", string0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<HashMap> class0 = HashMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      SimpleType simpleType0 = new SimpleType(mapType0);
      String string0 = simpleType0.getGenericSignature();
      assertEquals("Ljava/util/HashMap<Ljava/lang/Object;Ljava/lang/Object;>;", string0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Class<DecimalNode> class0 = DecimalNode.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      assertFalse(simpleType0.isArrayType());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_BOOL;
      boolean boolean0 = simpleType0.equals("Z;");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_CLASS;
      boolean boolean0 = simpleType0.equals(simpleType0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      boolean boolean0 = simpleType0.equals((Object) null);
      assertFalse(simpleType0.useStaticType());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_CLASS;
      SimpleType simpleType1 = simpleType0.withStaticTyping();
      boolean boolean0 = simpleType0.equals(simpleType1);
      assertTrue(boolean0);
      assertTrue(simpleType1.useStaticType());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_CLASS;
      SimpleType simpleType1 = TypeFactory.CORE_TYPE_BOOL;
      boolean boolean0 = simpleType0.equals(simpleType1);
      assertFalse(boolean0);
  }
}