/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 05:45:57 GMT 2023
 */

package com.fasterxml.jackson.databind.type;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.JsonAutoDetect;
import com.fasterxml.jackson.annotation.PropertyAccessor;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.Module;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.type.MapType;
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
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_INT;
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
  public void test01()  throws Throwable  {
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = new SimpleType(class0);
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
  public void test03()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_LONG;
      Class<Integer> class0 = Integer.class;
      TypeBindings typeBindings0 = TypeFactory.EMPTY_BINDINGS;
      JavaType[] javaTypeArray0 = new JavaType[4];
      javaTypeArray0[1] = (JavaType) simpleType0;
      ReferenceType referenceType0 = ReferenceType.construct((Class<?>) class0, typeBindings0, (JavaType) simpleType0, javaTypeArray0, javaTypeArray0[1]);
      StringBuilder stringBuilder0 = new StringBuilder("Gv^D~!Eqly#eN");
      referenceType0.getGenericSignature(stringBuilder0);
      assertEquals("Gv^D~!Eqly#eNLjava/lang/Integer<J;;", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Class<SimpleType> class0 = SimpleType.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      assertFalse(simpleType0.useStaticType());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_ENUM;
      String string0 = simpleType0.getErasedSignature();
      assertEquals("Ljava/lang/Enum;", string0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Class<Module> class0 = Module.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      assertFalse(simpleType0.isContainerType());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
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
  public void test08()  throws Throwable  {
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
  public void test09()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_ENUM;
      Class<String> class0 = String.class;
      JavaType javaType0 = simpleType0._narrow(class0);
      assertTrue(javaType0.isConcrete());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_STRING;
      Class<String> class0 = String.class;
      JavaType javaType0 = simpleType0._narrow(class0);
      assertSame(javaType0, simpleType0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_LONG;
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class0 = HashMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      MapType mapType1 = mapType0.withContentTypeHandler(simpleType0);
      assertTrue(mapType1.isMapLikeType());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_COMPARABLE;
      SimpleType simpleType1 = simpleType0.withTypeHandler((Object) null);
      assertSame(simpleType1, simpleType0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_STRING;
      SimpleType simpleType1 = simpleType0.withValueHandler(simpleType0);
      assertTrue(simpleType1.hasValueHandler());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_ENUM;
      SimpleType simpleType1 = simpleType0.withValueHandler((Object) null);
      assertSame(simpleType1, simpleType0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_ENUM;
      SimpleType simpleType1 = simpleType0.withStaticTyping();
      SimpleType simpleType2 = simpleType1.withStaticTyping();
      assertTrue(simpleType2.useStaticType());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      ObjectMapper objectMapper0 = new ObjectMapper();
      PropertyAccessor propertyAccessor0 = PropertyAccessor.FIELD;
      JsonAutoDetect.Visibility jsonAutoDetect_Visibility0 = JsonAutoDetect.Visibility.NON_PRIVATE;
      objectMapper0.setVisibility(propertyAccessor0, jsonAutoDetect_Visibility0);
      ObjectReader objectReader0 = objectMapper0.readerForUpdating(defaultSerializerProvider_Impl0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class0 = HashMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      StringBuilder stringBuilder0 = new StringBuilder("Gv^D~!Eqly#eN");
      SimpleType simpleType0 = new SimpleType(mapType0);
      simpleType0.getGenericSignature(stringBuilder0);
      assertEquals("Gv^D~!Eqly#eNLjava/util/HashMap<Ljava/lang/Object;Ljava/lang/Object;>;", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_COMPARABLE;
      boolean boolean0 = simpleType0.equals((Object) null);
      assertFalse(boolean0);
  }
}
