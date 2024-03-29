/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:41:18 GMT 2023
 */

package com.fasterxml.jackson.databind.type;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.ReferenceType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.util.HashMap;
import java.util.LinkedList;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class SimpleType_ESTest extends SimpleType_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      StringBuilder stringBuilder0 = new StringBuilder();
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_COMPARABLE;
      // Undeclared exception!
      try { 
        simpleType0.withContentValueHandler(stringBuilder0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Simple types have no content types; can not call withContenValueHandler()
         //
         verifyException("com.fasterxml.jackson.databind.type.SimpleType", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Class<SimpleType> class0 = SimpleType.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_ENUM;
      // Undeclared exception!
      try { 
        simpleType0.withContentTypeHandler(class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Simple types have no content types; can not call withContenTypeHandler()
         //
         verifyException("com.fasterxml.jackson.databind.type.SimpleType", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_INT;
      StringBuilder stringBuilder0 = new StringBuilder("QNVSAVVr(~QuUd?|,");
      stringBuilder0.append((Object) simpleType0);
      assertEquals("QNVSAVVr(~QuUd?|,[simple type, class int]", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Class<SimpleType> class0 = SimpleType.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      JavaType javaType0 = simpleType0._narrow(class0);
      assertSame(javaType0, simpleType0);
      assertFalse(javaType0.useStaticType());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      boolean boolean0 = simpleType0.isContainerType();
      assertFalse(simpleType0.useStaticType());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<LinkedList> class0 = LinkedList.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      assertEquals(1, collectionType0.containedTypeCount());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_ENUM;
      // Undeclared exception!
      try { 
        simpleType0.withContentType(simpleType0);
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
      Class<String> class0 = String.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      assertFalse(simpleType0.useStaticType());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      StringBuilder stringBuilder0 = new StringBuilder();
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_CLASS;
      StringBuilder stringBuilder1 = simpleType0.getErasedSignature(stringBuilder0);
      assertEquals("Ljava/lang/Class;", stringBuilder1.toString());
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
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_CLASS;
      Class<Object> class0 = Object.class;
      JavaType javaType0 = simpleType0._narrow(class0);
      assertTrue(javaType0.isJavaLangObject());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_INT;
      ReferenceType referenceType0 = new ReferenceType(simpleType0, simpleType0);
      ReferenceType referenceType1 = referenceType0.withContentTypeHandler(referenceType0);
      boolean boolean0 = referenceType1.equals(referenceType0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_BOOL;
      SimpleType simpleType1 = simpleType0.withTypeHandler((Object) null);
      assertSame(simpleType1, simpleType0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      StringBuilder stringBuilder0 = new StringBuilder();
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_INT;
      SimpleType simpleType1 = simpleType0.withValueHandler(stringBuilder0);
      assertTrue(simpleType1.hasValueHandler());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Class<String> class0 = String.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      SimpleType simpleType1 = simpleType0.withValueHandler((Object) null);
      assertSame(simpleType1, simpleType0);
      assertFalse(simpleType1.useStaticType());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_INT;
      SimpleType simpleType1 = simpleType0.withStaticTyping();
      SimpleType simpleType2 = simpleType1.withStaticTyping();
      assertTrue(simpleType2.useStaticType());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Class<HashMap> class0 = HashMap.class;
      Class<String> class1 = String.class;
      SimpleType simpleType0 = new SimpleType(class0);
      TypeBindings typeBindings0 = TypeBindings.create((Class<?>) class0, (JavaType) simpleType0, (JavaType) simpleType0);
      JavaType[] javaTypeArray0 = new JavaType[6];
      SimpleType simpleType1 = new SimpleType(class1, typeBindings0, simpleType0, javaTypeArray0);
      String string0 = simpleType1.buildCanonicalName();
      assertEquals("java.lang.String<java.util.HashMap,java.util.HashMap>", string0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Class<HashMap> class0 = HashMap.class;
      Class<String> class1 = String.class;
      SimpleType simpleType0 = new SimpleType(class0);
      TypeBindings typeBindings0 = TypeBindings.create((Class<?>) class0, (JavaType) simpleType0, (JavaType) simpleType0);
      JavaType[] javaTypeArray0 = new JavaType[6];
      SimpleType simpleType1 = new SimpleType(class1, typeBindings0, simpleType0, javaTypeArray0);
      StringBuilder stringBuilder0 = new StringBuilder("java.lang.String<java.util.HashMap,java.util.HashMap>");
      simpleType1.getGenericSignature(stringBuilder0);
      assertEquals("java.lang.String<java.util.HashMap,java.util.HashMap>Ljava/lang/String<Ljava/util/HashMap;Ljava/util/HashMap;>;", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_INT;
      Integer integer0 = new Integer((-1));
      boolean boolean0 = simpleType0.equals(integer0);
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
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_CLASS;
      boolean boolean0 = simpleType0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_INT;
      SimpleType simpleType1 = TypeFactory.CORE_TYPE_LONG;
      boolean boolean0 = simpleType1.equals(simpleType0);
      assertFalse(boolean0);
  }
}
