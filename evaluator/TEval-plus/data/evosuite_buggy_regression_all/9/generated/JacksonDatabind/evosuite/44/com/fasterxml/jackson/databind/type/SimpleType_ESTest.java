/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 05:47:40 GMT 2023
 */

package com.fasterxml.jackson.databind.type;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.Module;
import com.fasterxml.jackson.databind.node.LongNode;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.ReferenceType;
import com.fasterxml.jackson.databind.type.SimpleType;
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
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class0 = HashMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      SimpleType simpleType0 = new SimpleType(mapType0);
      String string0 = simpleType0.toString();
      assertEquals("[simple type, class java.util.HashMap<java.lang.Object,java.lang.Object>]", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Class<String> class0 = String.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      // Undeclared exception!
      try { 
        simpleType0.withContentValueHandler(class0);
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
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_STRING;
      Object object0 = new Object();
      // Undeclared exception!
      try { 
        simpleType0.withContentTypeHandler(object0);
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
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = new SimpleType(class0);
      boolean boolean0 = simpleType0.isContainerType();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_STRING;
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
  public void test05()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_CLASS;
      ReferenceType referenceType0 = new ReferenceType(simpleType0, simpleType0);
      ReferenceType referenceType1 = referenceType0.withContentValueHandler("Ljava/lang/Class;");
      assertFalse(referenceType1.isContainerType());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_STRING;
      String string0 = simpleType0.getErasedSignature();
      assertEquals("Ljava/lang/String;", string0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Class<String> class0 = String.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      SimpleType simpleType1 = simpleType0.withTypeHandler((Object) null);
      assertSame(simpleType1, simpleType0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
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
  public void test09()  throws Throwable  {
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
  public void test10()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_CLASS;
      Class<LongNode> class0 = LongNode.class;
      JavaType javaType0 = simpleType0._narrow(class0);
      assertFalse(javaType0.isFinal());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Class<LongNode> class0 = LongNode.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      JavaType javaType0 = simpleType0._narrow(class0);
      assertFalse(javaType0.useStaticType());
      assertSame(javaType0, simpleType0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_ENUM;
      Object object0 = new Object();
      SimpleType simpleType1 = simpleType0.withTypeHandler(object0);
      assertNotSame(simpleType1, simpleType0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_BOOL;
      SimpleType simpleType1 = simpleType0.withValueHandler((Object) null);
      assertSame(simpleType1, simpleType0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      SimpleType simpleType1 = simpleType0.withStaticTyping();
      SimpleType simpleType2 = simpleType1.withStaticTyping();
      assertTrue(simpleType2.useStaticType());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<LinkedList> class0 = LinkedList.class;
      Class<Module> class1 = Module.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class1);
      SimpleType simpleType0 = new SimpleType(collectionType0);
      StringBuilder stringBuilder0 = new StringBuilder("");
      simpleType0.getGenericSignature(stringBuilder0);
      assertEquals("Ljava/util/LinkedList<Lcom/fasterxml/jackson/databind/Module;>;", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      assertFalse(simpleType0.isArrayType());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Class<String> class0 = String.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      boolean boolean0 = simpleType0.equals("[simple type, class java.lang.String]");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_STRING;
      boolean boolean0 = simpleType0.equals(simpleType0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_STRING;
      boolean boolean0 = simpleType0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_COMPARABLE;
      SimpleType simpleType1 = simpleType0.withStaticTyping();
      boolean boolean0 = simpleType1.equals(simpleType0);
      assertTrue(simpleType1.useStaticType());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_STRING;
      SimpleType simpleType1 = TypeFactory.CORE_TYPE_LONG;
      boolean boolean0 = simpleType1.equals(simpleType0);
      assertFalse(boolean0);
  }
}