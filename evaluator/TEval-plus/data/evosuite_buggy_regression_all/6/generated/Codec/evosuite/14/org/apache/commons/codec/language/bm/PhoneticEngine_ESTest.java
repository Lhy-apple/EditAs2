/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:09:15 GMT 2023
 */

package org.apache.commons.codec.language.bm;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.LinkedList;
import java.util.Set;
import org.apache.commons.codec.language.bm.Languages;
import org.apache.commons.codec.language.bm.NameType;
import org.apache.commons.codec.language.bm.PhoneticEngine;
import org.apache.commons.codec.language.bm.Rule;
import org.apache.commons.codec.language.bm.RuleType;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class PhoneticEngine_ESTest extends PhoneticEngine_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      PhoneticEngine.PhonemeBuilder phoneticEngine_PhonemeBuilder0 = PhoneticEngine.PhonemeBuilder.empty((Languages.LanguageSet) null);
      Set<Rule.Phoneme> set0 = phoneticEngine_PhonemeBuilder0.getPhonemes();
      assertFalse(set0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      NameType nameType0 = NameType.SEPHARDIC;
      RuleType ruleType0 = RuleType.APPROX;
      PhoneticEngine phoneticEngine0 = null;
      try {
        phoneticEngine0 = new PhoneticEngine(nameType0, ruleType0, false);
        fail("Expecting exception: NoClassDefFoundError");
      
      } catch(NoClassDefFoundError e) {
         //
         // Could not initialize class org.apache.commons.codec.language.bm.Lang
         //
         verifyException("org.apache.commons.codec.language.bm.PhoneticEngine", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      PhoneticEngine.PhonemeBuilder phoneticEngine_PhonemeBuilder0 = PhoneticEngine.PhonemeBuilder.empty((Languages.LanguageSet) null);
      phoneticEngine_PhonemeBuilder0.append((CharSequence) null);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      PhoneticEngine.PhonemeBuilder phoneticEngine_PhonemeBuilder0 = PhoneticEngine.PhonemeBuilder.empty((Languages.LanguageSet) null);
      LinkedList<Rule.Phoneme> linkedList0 = new LinkedList<Rule.Phoneme>();
      Rule.PhonemeList rule_PhonemeList0 = new Rule.PhonemeList(linkedList0);
      phoneticEngine_PhonemeBuilder0.apply(rule_PhonemeList0, 0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      PhoneticEngine.PhonemeBuilder phoneticEngine_PhonemeBuilder0 = PhoneticEngine.PhonemeBuilder.empty((Languages.LanguageSet) null);
      Rule.Phoneme rule_Phoneme0 = new Rule.Phoneme("", (Languages.LanguageSet) null);
      // Undeclared exception!
      try { 
        phoneticEngine_PhonemeBuilder0.apply(rule_Phoneme0, 1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.codec.language.bm.PhoneticEngine$PhonemeBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      PhoneticEngine.PhonemeBuilder phoneticEngine_PhonemeBuilder0 = PhoneticEngine.PhonemeBuilder.empty((Languages.LanguageSet) null);
      String string0 = phoneticEngine_PhonemeBuilder0.makeString();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      NameType nameType0 = NameType.SEPHARDIC;
      RuleType ruleType0 = RuleType.RULES;
      PhoneticEngine phoneticEngine0 = null;
      try {
        phoneticEngine0 = new PhoneticEngine(nameType0, ruleType0, false);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // ruleType must not be RULES
         //
         verifyException("org.apache.commons.codec.language.bm.PhoneticEngine", e);
      }
  }
}