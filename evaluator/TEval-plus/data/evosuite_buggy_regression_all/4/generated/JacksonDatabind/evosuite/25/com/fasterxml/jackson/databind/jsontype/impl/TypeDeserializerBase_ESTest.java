/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:40:32 GMT 2023
 */

package com.fasterxml.jackson.databind.jsontype.impl;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.BeanProperty;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.jsontype.TypeIdResolver;
import com.fasterxml.jackson.databind.jsontype.impl.AsExternalTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.AsPropertyTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.ClassNameIdResolver;
import com.fasterxml.jackson.databind.jsontype.impl.MinimalClassNameIdResolver;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeFactory;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TypeDeserializerBase_ESTest extends TypeDeserializerBase_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      MinimalClassNameIdResolver minimalClassNameIdResolver0 = new MinimalClassNameIdResolver(simpleType0, typeFactory0);
      AsExternalTypeDeserializer asExternalTypeDeserializer0 = new AsExternalTypeDeserializer(simpleType0, minimalClassNameIdResolver0, "sM~0Z", false, class0);
      String string0 = asExternalTypeDeserializer0.getPropertyName();
      assertEquals("sM~0Z", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      MinimalClassNameIdResolver minimalClassNameIdResolver0 = new MinimalClassNameIdResolver(simpleType0, typeFactory0);
      AsExternalTypeDeserializer asExternalTypeDeserializer0 = new AsExternalTypeDeserializer(simpleType0, minimalClassNameIdResolver0, "sM~0Z", false, class0);
      MinimalClassNameIdResolver minimalClassNameIdResolver1 = (MinimalClassNameIdResolver)asExternalTypeDeserializer0.getTypeIdResolver();
      assertEquals("class name used as type id", minimalClassNameIdResolver1.getDescForKnownTypeIds());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Class<ClassNameIdResolver> class0 = ClassNameIdResolver.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      AsExternalTypeDeserializer asExternalTypeDeserializer0 = new AsExternalTypeDeserializer(simpleType0, (TypeIdResolver) null, "knwntype ids are not sttically known", false, class0);
      String string0 = asExternalTypeDeserializer0.toString();
      assertEquals("[com.fasterxml.jackson.databind.jsontype.impl.AsExternalTypeDeserializer; base-type:[simple type, class com.fasterxml.jackson.databind.jsontype.impl.ClassNameIdResolver]; id-resolver: null]", string0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Class<ClassNameIdResolver> class0 = ClassNameIdResolver.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      AsExternalTypeDeserializer asExternalTypeDeserializer0 = new AsExternalTypeDeserializer(simpleType0, (TypeIdResolver) null, "knwntype ids are not sttically known", true, class0);
      String string0 = asExternalTypeDeserializer0.baseTypeName();
      assertEquals("com.fasterxml.jackson.databind.jsontype.impl.ClassNameIdResolver", string0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Class<ClassNameIdResolver> class0 = ClassNameIdResolver.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      MinimalClassNameIdResolver minimalClassNameIdResolver0 = new MinimalClassNameIdResolver(simpleType0, typeFactory0);
      AsExternalTypeDeserializer asExternalTypeDeserializer0 = new AsExternalTypeDeserializer(simpleType0, minimalClassNameIdResolver0, "sM~0Z", false, class0);
      // Undeclared exception!
      try { 
        asExternalTypeDeserializer0._deserializeWithNativeTypeId((JsonParser) null, (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.impl.TypeDeserializerBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Class<ClassNameIdResolver> class0 = ClassNameIdResolver.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      MinimalClassNameIdResolver minimalClassNameIdResolver0 = new MinimalClassNameIdResolver(simpleType0, typeFactory0);
      AsExternalTypeDeserializer asExternalTypeDeserializer0 = new AsExternalTypeDeserializer(simpleType0, minimalClassNameIdResolver0, "sM~0Z", false, class0);
      AsExternalTypeDeserializer asExternalTypeDeserializer1 = new AsExternalTypeDeserializer(asExternalTypeDeserializer0, (BeanProperty) null);
      assertFalse(asExternalTypeDeserializer1.equals((Object)asExternalTypeDeserializer0));
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Class<ClassNameIdResolver> class0 = ClassNameIdResolver.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      AsExternalTypeDeserializer asExternalTypeDeserializer0 = new AsExternalTypeDeserializer(simpleType0, (TypeIdResolver) null, "No (native) type id found when one was expected for polymorphic type handling", true, (Class<?>) null);
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        asExternalTypeDeserializer0._deserializeWithNativeTypeId((JsonParser) null, defaultDeserializationContext_Impl0, (Object) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.NullifyingDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Class<ClassNameIdResolver> class0 = ClassNameIdResolver.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      MinimalClassNameIdResolver minimalClassNameIdResolver0 = new MinimalClassNameIdResolver(simpleType0, typeFactory0);
      JsonTypeInfo.As jsonTypeInfo_As0 = JsonTypeInfo.As.EXISTING_PROPERTY;
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(simpleType0, minimalClassNameIdResolver0, (String) null, true, class0, jsonTypeInfo_As0);
      Class<?> class1 = asPropertyTypeDeserializer0.getDefaultImpl();
      assertFalse(class1.isInterface());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Class<ClassNameIdResolver> class0 = ClassNameIdResolver.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(simpleType0, (TypeIdResolver) null, "known type ids are not statically known", true, (Class<?>) null);
      Class<?> class1 = asPropertyTypeDeserializer0.getDefaultImpl();
      assertNull(class1);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Class<ClassNameIdResolver> class0 = ClassNameIdResolver.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      AsExternalTypeDeserializer asExternalTypeDeserializer0 = new AsExternalTypeDeserializer(simpleType0, (TypeIdResolver) null, "No (native) type id found when one was expected for polymorphic type handling", true, class0);
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        asExternalTypeDeserializer0._deserializeWithNativeTypeId((JsonParser) null, defaultDeserializationContext_Impl0, "No (native) type id found when one was expected for polymorphic type handling");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.impl.TypeDeserializerBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Class<ClassNameIdResolver> class0 = ClassNameIdResolver.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      AsExternalTypeDeserializer asExternalTypeDeserializer0 = new AsExternalTypeDeserializer(simpleType0, (TypeIdResolver) null, "knwntype ids are not sttically known", false, class0);
      // Undeclared exception!
      try { 
        asExternalTypeDeserializer0._deserializeWithNativeTypeId((JsonParser) null, (DeserializationContext) null, (Object) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.impl.TypeDeserializerBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      AsExternalTypeDeserializer asExternalTypeDeserializer0 = new AsExternalTypeDeserializer(simpleType0, (TypeIdResolver) null, "0n)C?T'LS33YOO", false, class0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        asExternalTypeDeserializer0._deserializeWithNativeTypeId((JsonParser) null, defaultDeserializationContext_Impl0, defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.impl.TypeDeserializerBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Class<ClassNameIdResolver> class0 = ClassNameIdResolver.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      AsExternalTypeDeserializer asExternalTypeDeserializer0 = new AsExternalTypeDeserializer(simpleType0, (TypeIdResolver) null, "knwntype ids are not sttically known", true, class0);
      // Undeclared exception!
      try { 
        asExternalTypeDeserializer0._handleUnknownTypeId((DeserializationContext) null, "knwntype ids are not sttically known", (TypeIdResolver) null, simpleType0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.impl.TypeDeserializerBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      MinimalClassNameIdResolver minimalClassNameIdResolver0 = new MinimalClassNameIdResolver(simpleType0, typeFactory0);
      JsonTypeInfo.As jsonTypeInfo_As0 = JsonTypeInfo.As.EXISTING_PROPERTY;
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(simpleType0, minimalClassNameIdResolver0, (String) null, true, class0, jsonTypeInfo_As0);
      // Undeclared exception!
      try { 
        asPropertyTypeDeserializer0._handleUnknownTypeId((DeserializationContext) null, (String) null, minimalClassNameIdResolver0, simpleType0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.impl.TypeDeserializerBase", e);
      }
  }
}
